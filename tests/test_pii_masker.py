"""
Unit tests for the PII Masking layer (Phase 5).

Tests verify:
1. Kenya 8-digit ID numbers are redacted
2. Uganda 14-character National IDs are redacted
3. Nigeria 11-digit NINs are redacted
4. African phone numbers are redacted (with and without country code)
5. Email addresses are redacted
6. Person names detected by spaCy are redacted
7. Turaco policy numbers are NOT redacted (needed for processing)
8. mask_dict redacts patient_name but preserves all other fields
9. mask_receipt_result wraps correctly without mutating the original
10. Empty and None inputs are handled safely
"""

import pytest
from app.modules.pii_masker import mask_text, mask_dict, mask_receipt_result


# ── Regex PII tests ────────────────────────────────────────────────────────────

def test_kenya_id_redacted():
    text = "Patient ID: 34521876 was admitted on Monday."
    result = mask_text(text)
    assert "34521876" not in result
    assert "[REDACTED ID]" in result


def test_uganda_id_redacted():
    text = "National ID: CM91001000097RE presented at reception."
    result = mask_text(text)
    assert "CM91001000097RE" not in result
    assert "[REDACTED ID]" in result


def test_nigeria_nin_redacted():
    text = "NIN: 12345678901 verified at hospital gate."
    result = mask_text(text)
    assert "12345678901" not in result
    assert "[REDACTED ID]" in result


def test_phone_with_country_code_redacted():
    text = "Call patient on +254712345678 after discharge."
    result = mask_text(text)
    assert "712345678" not in result
    assert "[REDACTED PHONE]" in result


def test_phone_local_format_redacted():
    text = "Emergency contact: 0712345678"
    result = mask_text(text)
    assert "0712345678" not in result
    assert "[REDACTED PHONE]" in result


def test_email_redacted():
    text = "Send report to john.kamau@gmail.com for follow-up."
    result = mask_text(text)
    assert "john.kamau@gmail.com" not in result
    assert "[REDACTED EMAIL]" in result


def test_multiple_pii_types_in_one_text():
    text = (
        "Patient John Kamau (ID: 34521876) can be reached at "
        "+254712345678 or john@email.com. Policy: TUR-KE-HC-001-88234"
    )
    result = mask_text(text)
    assert "34521876" not in result
    assert "712345678" not in result
    assert "john@email.com" not in result


# ── Policy number preservation ─────────────────────────────────────────────────

def test_policy_number_not_redacted():
    text = "Policy TUR-KE-HC-001-88234 is active and valid."
    result = mask_text(text)
    assert "TUR-KE-HC-001-88234" in result, "Policy numbers must NOT be redacted"


# ── Contextual name detection (regex-based) ───────────────────────────────────

def test_labelled_patient_name_redacted():
    text = "Patient Name: John Kamau Mwangi was admitted on Monday."
    result = mask_text(text)
    assert "John Kamau Mwangi" not in result
    assert "[REDACTED NAME]" in result


def test_titled_name_redacted():
    text = "Authorized By: Dr. Alice Njoroge"
    result = mask_text(text)
    assert "Alice Njoroge" not in result
    assert "[REDACTED NAME]" in result


def test_standalone_doctor_title_redacted():
    text = "Attending physician Dr. Samuel Kariuki signed the discharge."
    result = mask_text(text)
    assert "Samuel Kariuki" not in result


def test_hospital_name_not_redacted():
    text = "Kenyatta National Hospital, Nairobi, Kenya."
    result = mask_text(text)
    assert "Kenyatta National Hospital" in result
    assert "Nairobi" in result


# ── mask_dict tests ────────────────────────────────────────────────────────────

def test_mask_dict_redacts_patient_name():
    data = {
        "patient_name": "John Kamau Mwangi",
        "hospital_name": "Kenyatta National Hospital",
        "diagnosis": "Malaria",
        "nights": 3,
        "total_cost": 6000.0,
        "country": "Kenya",
        "policy_number": "TUR-KE-HC-001-88234",
    }
    masked = mask_dict(data)
    assert masked["patient_name"] == "[REDACTED NAME]"


def test_mask_dict_preserves_all_other_fields():
    data = {
        "patient_name": "Grace Otieno",
        "hospital_name": "Nairobi Hospital",
        "diagnosis": "Pneumonia",
        "diagnosis_code": "J18",
        "nights": 15,
        "total_cost": 40000.0,
        "currency": "KES",
        "country": "Kenya",
        "policy_number": "TUR-KE-HC-001-71092",
        "admission_date": "2026-03-10",
        "discharge_date": "2026-03-25",
    }
    masked = mask_dict(data)
    assert masked["hospital_name"] == "Nairobi Hospital"
    assert masked["diagnosis"] == "Pneumonia"
    assert masked["diagnosis_code"] == "J18"
    assert masked["nights"] == 15
    assert masked["total_cost"] == 40000.0
    assert masked["policy_number"] == "TUR-KE-HC-001-71092"
    assert masked["country"] == "Kenya"


def test_mask_dict_does_not_mutate_original():
    data = {"patient_name": "Alice Njoroge", "diagnosis": "Typhoid"}
    original_name = data["patient_name"]
    mask_dict(data)
    assert data["patient_name"] == original_name, "mask_dict must not mutate the input dict"


def test_mask_dict_handles_none_patient_name():
    data = {"patient_name": None, "diagnosis": "Malaria"}
    masked = mask_dict(data)
    assert masked["patient_name"] is None  # None stays None — nothing to redact


# ── mask_receipt_result tests ──────────────────────────────────────────────────

def test_mask_receipt_result_structure():
    result = {
        "data": {"patient_name": "James Omondi", "diagnosis": "Typhoid"},
        "confidence": 0.85,
        "method": "llava",
    }
    masked = mask_receipt_result(result)
    assert masked["data"]["patient_name"] == "[REDACTED NAME]"
    assert masked["data"]["diagnosis"] == "Typhoid"
    assert masked["confidence"] == 0.85
    assert masked["method"] == "llava"


def test_mask_receipt_result_does_not_mutate_original():
    result = {
        "data": {"patient_name": "James Omondi"},
        "confidence": 0.85,
        "method": "llava",
    }
    mask_receipt_result(result)
    assert result["data"]["patient_name"] == "James Omondi"


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_empty_string_handled():
    assert mask_text("") == ""


def test_whitespace_only_handled():
    result = mask_text("   ")
    assert result.strip() == ""


def test_text_with_no_pii_unchanged_structure():
    text = "Diagnosis: Malaria. Admitted for 3 nights. Total: KES 6000."
    result = mask_text(text)
    assert "Malaria" in result
    assert "KES 6000" in result
