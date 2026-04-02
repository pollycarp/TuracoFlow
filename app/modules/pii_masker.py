"""
PII Masking Layer (Phase 5)

Redacts personally identifiable information from receipt text and structured
data before it is processed by any LLM or stored in logs.

Design choice: uses regex-only (no external NLP model) for maximum portability
and reliability. Hospital receipts are structured documents — names always
appear after known labels like "Patient Name:" or "Patient:", making targeted
regex more reliable than a general NER model for this specific use case.

Two modes:
  mask_text(text)  — redacts PII from a raw string (used before LLM calls)
  mask_dict(data)  — redacts PII fields from extracted receipt dict

PII types handled:
  - Person names after known labels → [REDACTED NAME]  (contextual regex)
  - National ID numbers             → [REDACTED ID]    (regex: KE/UG/NG formats)
  - Phone numbers                   → [REDACTED PHONE] (regex: African mobile formats)
  - Email addresses                 → [REDACTED EMAIL] (regex)
"""

import logging
import re

logger = logging.getLogger(__name__)

# ── Name patterns (contextual — anchored to known receipt labels) ──────────────
# Matches "Patient Name: John Kamau Mwangi" style entries
# Captures 2–4 capitalised words following a known label
_NAME_LABEL_PATTERN = re.compile(
    r'(?P<label>'
    r'Patient\s*Name|Patient|Name|Claimant|Beneficiary|Policyholder|'
    r'Authorized\s*By|Attending|Dr\.|Doctor'
    r')\s*[:\-]\s*'
    r'(?P<name>(?:Dr\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
    re.IGNORECASE,
)

# Standalone "Dr. Firstname Lastname" pattern (not preceded by a label)
_TITLE_NAME_PATTERN = re.compile(
    r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b'
)

# ── ID number patterns ─────────────────────────────────────────────────────────

# Uganda National ID: 2 letters + 11 digits + 2 letters (e.g. CM91001000097RE)
# Must be checked before the 11-digit Nigeria NIN pattern to avoid conflict
_RE_UG_ID = re.compile(r'\b[A-Z]{2}\d{11}[A-Z]{2}\b')

# Nigeria NIN: exactly 11 digits
_RE_NG_NIN = re.compile(r'\b\d{11}\b')

# Kenya National ID: 7–8 digits
_RE_KE_ID = re.compile(r'\b\d{7,8}\b')

# ── Phone numbers ──────────────────────────────────────────────────────────────
# African formats: +254 / +256 / +234 country codes, or local 07xx/08xx/09xx
_RE_PHONE = re.compile(
    r'(?:\+?(?:254|256|234|255|233|260)\s?)?'
    r'(?:0)?[7-9]\d{8}\b'
)

# ── Email addresses ────────────────────────────────────────────────────────────
_RE_EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b')

# ── Policy numbers — explicitly preserved (needed for claim lookup) ────────────
# Turaco format: TUR-KE-HC-001-XXXXX
# We store them before masking and restore them after to be safe
_RE_POLICY = re.compile(r'\bTUR-[A-Z]{2}-[A-Z]{2,3}-\d{3}-\d+\b')


# ── Core masking helpers ───────────────────────────────────────────────────────

def _preserve_and_restore(text: str, pattern: re.Pattern, placeholder: str):
    """
    Temporarily replace matched tokens with a placeholder so subsequent
    regex passes don't accidentally clobber them.
    Returns (modified_text, restore_map).
    """
    restore_map = {}
    def replacer(m):
        key = f"__{placeholder}_{len(restore_map)}__"
        restore_map[key] = m.group(0)
        return key
    return pattern.sub(replacer, text), restore_map


def _restore(text: str, restore_map: dict) -> str:
    for key, value in restore_map.items():
        text = text.replace(key, value)
    return text


def _mask_names(text: str) -> str:
    """Redact names using contextual label-anchored regex patterns."""
    # Mask label-anchored names (e.g. "Patient Name: John Kamau")
    def replace_labelled(m):
        label = m.group("label")
        return f"{label}: [REDACTED NAME]"

    text = _NAME_LABEL_PATTERN.sub(replace_labelled, text)

    # Mask titled names (e.g. "Dr. Alice Njoroge")
    text = _TITLE_NAME_PATTERN.sub("[REDACTED NAME]", text)

    return text


def mask_text(text: str) -> str:
    """
    Mask all PII in a raw text string.
    Used before sending OCR text to an LLM.

    Order:
      1. Preserve policy numbers (must survive the ID digit patterns)
      2. Regex: IDs, phones, emails
      3. Contextual name patterns
      4. Restore policy numbers

    Returns the masked string.
    """
    if not text or not text.strip():
        return text

    # Step 1: Protect policy numbers from digit-based ID patterns
    text, policy_map = _preserve_and_restore(text, _RE_POLICY, "POLICY")

    # Step 2: Mask IDs (order: longest/most specific first)
    text = _RE_UG_ID.sub("[REDACTED ID]", text)
    text = _RE_NG_NIN.sub("[REDACTED ID]", text)
    text = _RE_KE_ID.sub("[REDACTED ID]", text)

    # Step 3: Mask phones and emails
    text = _RE_PHONE.sub("[REDACTED PHONE]", text)
    text = _RE_EMAIL.sub("[REDACTED EMAIL]", text)

    # Step 4: Mask names
    text = _mask_names(text)

    # Step 5: Restore policy numbers
    text = _restore(text, policy_map)

    return text


def mask_dict(data: dict) -> dict:
    """
    Mask PII fields in a structured extracted receipt dict.
    Returns a new dict — does not mutate the original.

    Fields masked:
      - patient_name → "[REDACTED NAME]"

    Fields kept as-is (needed for claim processing):
      hospital_name, diagnosis, diagnosis_code, total_cost,
      currency, country, nights, dates, policy_number
    """
    masked = dict(data)
    if masked.get("patient_name"):
        masked["patient_name"] = "[REDACTED NAME]"
    return masked


def mask_receipt_result(result: dict) -> dict:
    """
    Mask PII in a full extraction result dict (as returned by extractor.extract_receipt).
    Returns a new result dict — does not mutate the original.
    """
    return {
        **result,
        "data": mask_dict(result.get("data", {})),
    }
