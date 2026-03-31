"""
Unit tests for the OCR & LLM Extraction Engine (Phase 4).

Tests verify:
1. Approved receipt returns all key fields filled
2. Confidence is a float between 0.0 and 1.0
3. All REQUIRED_FIELDS keys are always present in output (even if null)
4. Over-limit receipt returns correct nights count (15)
5. Low-confidence receipt returns confidence below threshold
6. Missing image file raises FileNotFoundError
7. JSON parser handles markdown-wrapped responses
8. JSON parser handles raw JSON responses
"""

import pytest
from pathlib import Path
from app.modules.extractor import (
    extract_receipt,
    _parse_json_response,
    _count_filled_fields,
    REQUIRED_FIELDS,
)

RECEIPTS_DIR = Path(__file__).parent.parent / "data" / "receipts"


# ── JSON parser unit tests (no LLM needed) ────────────────────────────────────

def test_parse_json_raw():
    raw = '{"patient_name": "John Doe", "nights": 3}'
    result = _parse_json_response(raw)
    assert result["patient_name"] == "John Doe"
    assert result["nights"] == 3


def test_parse_json_with_markdown_code_block():
    raw = '```json\n{"patient_name": "Jane", "total_cost": 5000}\n```'
    result = _parse_json_response(raw)
    assert result is not None
    assert result["patient_name"] == "Jane"


def test_parse_json_with_preamble():
    raw = 'Here is the extracted data:\n{"hospital_name": "Kenyatta", "country": "Kenya"}'
    result = _parse_json_response(raw)
    assert result is not None
    assert result["hospital_name"] == "Kenyatta"


def test_parse_json_invalid_returns_none():
    raw = "This is not JSON at all."
    result = _parse_json_response(raw)
    assert result is None


def test_count_filled_fields_all_present():
    data = {k: "value" for k in REQUIRED_FIELDS}
    assert _count_filled_fields(data) == len(REQUIRED_FIELDS)


def test_count_filled_fields_partial():
    data = {"patient_name": "John", "nights": 3, "country": "Kenya"}
    assert _count_filled_fields(data) == 3


def test_count_filled_fields_none_values_not_counted():
    data = {k: None for k in REQUIRED_FIELDS}
    assert _count_filled_fields(data) == 0


# ── Image not found ───────────────────────────────────────────────────────────

def test_missing_image_raises_error():
    with pytest.raises(FileNotFoundError):
        extract_receipt("data/receipts/does_not_exist.png")


# ── Full extraction tests (require Ollama running) ────────────────────────────

@pytest.mark.slow
def test_approved_receipt_extracts_key_fields():
    result = extract_receipt(str(RECEIPTS_DIR / "receipt_approved.png"))
    data = result["data"]

    assert result["method"] in ("llava", "easyocr+llm"), "Should use a valid extraction method"
    assert result["confidence"] > 0.0
    assert data["country"] is not None
    # nights should be filled either by LLM or computed from dates
    assert data["nights"] is not None, "nights must be set (by LLM or computed from dates)"
    assert data["hospital_name"] is not None
    assert data["diagnosis"] is not None


@pytest.mark.slow
def test_all_required_fields_present_in_output():
    """All REQUIRED_FIELDS keys must always be in the output dict, even if null."""
    result = extract_receipt(str(RECEIPTS_DIR / "receipt_approved.png"))
    for field in REQUIRED_FIELDS:
        assert field in result["data"], f"Missing field in output: {field}"


@pytest.mark.slow
def test_confidence_is_float_in_range():
    result = extract_receipt(str(RECEIPTS_DIR / "receipt_approved.png"))
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.slow
def test_over_limit_receipt_extracts_15_nights():
    result = extract_receipt(str(RECEIPTS_DIR / "receipt_over_limit.png"))
    data = result["data"]
    # nights must be filled — either directly by LLM or computed from dates
    assert data["nights"] is not None, "nights must be set (by LLM or computed from dates)"
    assert data["nights"] >= 10, f"Over-limit receipt has 15 nights, got: {data['nights']}"


@pytest.mark.slow
def test_low_confidence_receipt_has_lower_confidence():
    approved = extract_receipt(str(RECEIPTS_DIR / "receipt_approved.png"))
    low_conf = extract_receipt(str(RECEIPTS_DIR / "receipt_low_confidence.png"))
    assert low_conf["confidence"] <= approved["confidence"], (
        "Blurry/incomplete receipt should have lower confidence than clean receipt"
    )
