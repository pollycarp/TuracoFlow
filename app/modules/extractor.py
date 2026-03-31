"""
OCR & LLM Extraction Engine (Phase 4)

Takes a receipt image and returns structured claim data.

Two-path strategy:
  Primary path:  LLaVA (vision LLM via Ollama) reads the image directly.
  Fallback path: EasyOCR extracts raw text → LLaMA 3.2 structures it into JSON.

The fallback kicks in automatically if:
  - LLaVA returns incomplete fields (< MIN_REQUIRED_FIELDS filled)
  - LLaVA response cannot be parsed as valid JSON
"""

import json
import logging
import re
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional

import easyocr
import numpy as np
import ollama
from PIL import Image, ImageFilter

from app.core.config import settings

logger = logging.getLogger(__name__)

# Lazy singleton — EasyOCR takes ~10s to initialise; we only want to do it once
_easyocr_reader: Optional[easyocr.Reader] = None


def _get_ocr_reader() -> easyocr.Reader:
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Initialising EasyOCR reader (first-time load)...")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def _ocr_raw_text(image_path: str) -> str:
    """Extract raw text from image using the cached EasyOCR reader."""
    reader = _get_ocr_reader()
    results = reader.readtext(image_path, detail=0)
    return "\n".join(results)

# Fields we must extract from every receipt
REQUIRED_FIELDS = [
    "patient_name",
    "hospital_name",
    "admission_date",
    "discharge_date",
    "nights",
    "diagnosis",
    "diagnosis_code",
    "total_cost",
    "currency",
    "country",
    "policy_number",
]

# Minimum number of required fields that must be present to trust LLaVA result
MIN_REQUIRED_FIELDS = 6

EXTRACTION_PROMPT = """You are a medical claims processor for Turaco Insurance in Africa.
Extract the following fields from this hospital receipt image.

Return ONLY valid JSON with these exact keys (use null for missing fields):
{
  "patient_name": "full name of patient",
  "hospital_name": "name of hospital or clinic",
  "admission_date": "date admitted (YYYY-MM-DD format)",
  "discharge_date": "date discharged (YYYY-MM-DD format)",
  "nights": <integer number of nights stayed, or null>,
  "diagnosis": "primary diagnosis in plain English",
  "diagnosis_code": "ICD-10 code if visible, else null",
  "total_cost": <numeric amount as float, no currency symbol>,
  "currency": "3-letter currency code e.g. KES, UGX, NGN",
  "country": "Kenya or Uganda or Nigeria or Unknown",
  "policy_number": "Turaco policy number if visible, else null"
}

Rules:
- Return ONLY the JSON object. No explanation, no markdown, no code blocks.
- If a field is not visible or unclear, use null.
- For nights: calculate from admission and discharge dates if not explicitly stated.
- For currency: infer from country or amounts shown (KES=Kenya, UGX=Uganda, NGN=Nigeria).
"""

TEXT_STRUCTURING_PROMPT = """You are a medical claims processor for Turaco Insurance in Africa.
Structure the following raw OCR text from a hospital receipt into JSON.

RAW OCR TEXT:
{ocr_text}

Return ONLY valid JSON with these exact keys (use null for missing fields):
{{
  "patient_name": "full name of patient",
  "hospital_name": "name of hospital or clinic",
  "admission_date": "date admitted (YYYY-MM-DD format)",
  "discharge_date": "date discharged (YYYY-MM-DD format)",
  "nights": <integer number of nights stayed, or null>,
  "diagnosis": "primary diagnosis in plain English",
  "diagnosis_code": "ICD-10 code if visible, else null",
  "total_cost": <numeric amount as float, no currency symbol>,
  "currency": "3-letter currency code e.g. KES, UGX, NGN",
  "country": "Kenya or Uganda or Nigeria or Unknown",
  "policy_number": "Turaco policy number if visible, else null"
}}

Rules:
- Return ONLY the JSON object. No explanation, no markdown, no code blocks.
- If a field is not visible or unclear, use null.
- For nights: calculate from admission and discharge dates if not explicitly stated.
"""


def _count_filled_fields(data: dict) -> int:
    """Count how many required fields are non-null."""
    return sum(1 for k in REQUIRED_FIELDS if data.get(k) is not None)


def _parse_json_response(text: str) -> Optional[dict]:
    """
    Safely parse JSON from LLM response.
    Handles cases where the model wraps JSON in markdown code blocks.
    """
    text = text.strip()

    # Strip markdown code blocks if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e} | Raw: {text[:200]}")
        return None


def _image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for Ollama vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_via_llava(image_path: str) -> tuple[Optional[dict], float]:
    """
    Primary path: use LLaVA vision model to read the receipt image directly.

    Returns:
        (extracted_data, confidence_score)
        confidence_score is between 0.0 and 1.0
    """
    try:
        image_b64 = _image_to_base64(image_path)

        response = ollama.chat(
            model=settings.ollama_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT,
                    "images": [image_b64],
                }
            ],
        )

        raw_text = response["message"]["content"]
        logger.debug(f"LLaVA raw response: {raw_text[:300]}")

        data = _parse_json_response(raw_text)
        if data is None:
            return None, 0.0

        filled = _count_filled_fields(data)
        confidence = round(filled / len(REQUIRED_FIELDS), 2)
        logger.info(f"LLaVA extracted {filled}/{len(REQUIRED_FIELDS)} fields (confidence={confidence})")
        return data, confidence

    except Exception as e:
        logger.error(f"LLaVA extraction failed: {e}")
        return None, 0.0


def _extract_via_easyocr_and_llm(
    image_path: str, ocr_text: Optional[str] = None
) -> tuple[Optional[dict], float]:
    """
    Fallback path: EasyOCR extracts raw text → LLaMA 3.2 structures it.
    Pass ocr_text to reuse already-extracted text and skip a second OCR run.

    Returns:
        (extracted_data, confidence_score)
    """
    try:
        # Step 1: EasyOCR — extract raw text (skip if already provided)
        if ocr_text is None:
            logger.info("Fallback: running EasyOCR on image...")
            ocr_text = _ocr_raw_text(image_path)

        if not ocr_text.strip():
            logger.warning("EasyOCR returned empty text")
            return None, 0.0

        logger.debug(f"EasyOCR raw text ({len(ocr_text)} chars): {ocr_text[:200]}")

        # Step 2: LLaMA 3.2 — structure the raw text
        prompt = TEXT_STRUCTURING_PROMPT.format(ocr_text=ocr_text)

        response = ollama.chat(
            model=settings.ollama_text_model,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response["message"]["content"]
        data = _parse_json_response(raw_text)

        if data is None:
            return None, 0.0

        filled = _count_filled_fields(data)
        # Fallback path gets a lower baseline confidence since OCR may have errors
        confidence = round((filled / len(REQUIRED_FIELDS)) * 0.85, 2)
        logger.info(f"Fallback extracted {filled}/{len(REQUIRED_FIELDS)} fields (confidence={confidence})")
        return data, confidence

    except Exception as e:
        logger.error(f"EasyOCR/LLM fallback failed: {e}")
        return None, 0.0


def _image_blur_score(image_path: str) -> float:
    """
    Measure image sharpness using Laplacian variance.
    Returns a score from 0.0 (very blurry) to 1.0 (sharp).

    Threshold tuned for typical receipt photos:
      variance < 100  → very blurry (score near 0)
      variance > 1000 → sharp (score near 1)
    """
    img = Image.open(image_path).convert("L")  # grayscale
    arr = np.array(img, dtype=np.float32)
    laplacian = np.array(img.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    variance = float(np.var(laplacian))
    # Normalise to [0, 1] with soft cap at 1000
    score = min(variance / 1000.0, 1.0)
    logger.debug(f"Blur score: {score:.3f} (variance={variance:.1f})")
    return round(score, 3)


DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d %B %Y",
    "%B %d, %Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%b %d, %Y",
]


def _parse_date_flexible(date_str: str) -> Optional[datetime]:
    """Try multiple date formats to parse a date string from LLM output."""
    if not date_str:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _rescue_nights_from_text(raw_text: str) -> Optional[int]:
    """
    Regex-based rescue: find 'Nights: X' or 'X nights' patterns in OCR text.
    Used when the LLM fails to extract nights reliably.
    """
    text = raw_text.lower()
    patterns = [
        r'nights?\s*[:\-]\s*(\d+)',
        r'(\d+)\s*nights?\b',
        r'no\.?\s*of\s*nights?\s*[:\-]\s*(\d+)',
        r'length\s+of\s+stay\s*[:\-]\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            logger.info(f"Regex rescued nights={value} using pattern '{pattern}'")
            return value
    return None


def _infer_nights(data: dict, raw_text: str = "") -> dict:
    """
    Fill in missing nights using two strategies (in order):
      1. Compute from admission_date and discharge_date (handles multiple formats)
      2. Regex rescue from raw OCR text
    Mutates and returns the data dict.
    """
    if data.get("nights") is not None:
        return data

    # Strategy 1: compute from dates
    admit = _parse_date_flexible(data.get("admission_date") or "")
    discharge = _parse_date_flexible(data.get("discharge_date") or "")
    if admit and discharge:
        computed = (discharge - admit).days
        if computed > 0:
            data["nights"] = computed
            logger.info(f"Computed nights from dates: {computed}")
            return data

    # Strategy 2: regex on raw OCR text
    if raw_text:
        rescued = _rescue_nights_from_text(raw_text)
        if rescued is not None:
            data["nights"] = rescued

    return data


def extract_receipt(image_path: str) -> dict:
    """
    Main entry point. Extract structured data from a receipt image.

    Strategy:
    1. Try LLaVA (vision model) first
    2. If LLaVA returns < MIN_REQUIRED_FIELDS filled, fall back to EasyOCR + LLaMA
    3. If both fail, return empty structure with confidence=0.0

    Returns a dict with keys:
        data        — the extracted fields (all REQUIRED_FIELDS keys present)
        confidence  — float 0.0–1.0
        method      — "llava" | "easyocr+llm" | "failed"
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Receipt image not found: {image_path}")

    logger.info(f"Extracting receipt: {image_path}")

    # Score image sharpness — used to penalise confidence for blurry photos
    blur_score = _image_blur_score(image_path)
    logger.info(f"Image blur score: {blur_score}")

    # --- Primary: LLaVA ---
    data, confidence = _extract_via_llava(image_path)

    if data is not None and _count_filled_fields(data) >= MIN_REQUIRED_FIELDS:
        # Try date-based calculation first; only run OCR if nights still missing
        data = _infer_nights(data)
        if data.get("nights") is None:
            logger.info("nights still None after date inference — running OCR rescue...")
            raw_text = _ocr_raw_text(image_path)
            data = _infer_nights(data, raw_text)
        adjusted = round(confidence * blur_score, 2)
        return _build_result(data, adjusted, method="llava")

    logger.info(f"LLaVA result insufficient ({_count_filled_fields(data or {})} fields). Trying fallback...")

    # --- Fallback: EasyOCR + LLaMA ---
    # _extract_via_easyocr_and_llm already runs OCR internally; capture the text
    # for _infer_nights reuse by running it here and passing it down
    raw_text = _ocr_raw_text(image_path)
    fallback_data, fallback_confidence = _extract_via_easyocr_and_llm(image_path, ocr_text=raw_text)

    if fallback_data is not None:
        fallback_data = _infer_nights(fallback_data, raw_text)
        adjusted = round(fallback_confidence * blur_score, 2)
        return _build_result(fallback_data, adjusted, method="easyocr+llm")

    # --- Both failed ---
    logger.error("Both extraction paths failed.")
    return _build_result({}, 0.0, method="failed")


def _build_result(data: dict, confidence: float, method: str) -> dict:
    """Ensure all required fields are present in the output (null if missing)."""
    normalized = {field: data.get(field) for field in REQUIRED_FIELDS}
    return {
        "data": normalized,
        "confidence": confidence,
        "method": method,
    }
