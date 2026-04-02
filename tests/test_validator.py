"""
Unit tests for Claims Validation Logic (Phase 6).

Tests verify:
1. Valid claim → APPROVED with correct payout amount
2. Low confidence → REVIEW (not REJECTED)
3. Over-limit nights → REJECTED
4. Untrusted hospital → REJECTED
5. Excluded diagnosis → REJECTED
6. Missing critical fields → REJECTED
7. Hospital fuzzy matching works
8. to_dict() output has the right keys
9. Uganda and Nigeria policies validate correctly
"""

import pytest
from app.modules.validator import ClaimsValidator, _find_hospital, _load_trusted_hospitals


@pytest.fixture(scope="module")
def validator():
    return ClaimsValidator()


@pytest.fixture(scope="module")
def hospitals():
    return _load_trusted_hospitals()


# ── Hospital matching ──────────────────────────────────────────────────────────

def test_exact_hospital_match(hospitals):
    result = _find_hospital("Kenyatta National Hospital", hospitals)
    assert result is not None
    assert result["country"] == "Kenya"


def test_fuzzy_hospital_match(hospitals):
    # Abbreviated name — should still match
    result = _find_hospital("Kenyatta National Hosp", hospitals)
    assert result is not None


def test_unknown_hospital_returns_none(hospitals):
    result = _find_hospital("Random Unregistered Clinic", hospitals)
    assert result is None


def test_uganda_hospital_match(hospitals):
    result = _find_hospital("Mulago National Referral Hospital", hospitals)
    assert result is not None
    assert result["country"] == "Uganda"


def test_nigeria_hospital_match(hospitals):
    result = _find_hospital("Lagos University Teaching Hospital", hospitals)
    assert result is not None
    assert result["country"] == "Nigeria"


# ── Confidence check ───────────────────────────────────────────────────────────

def test_low_confidence_returns_review(validator):
    data = {
        "hospital_name": "Kenyatta National Hospital",
        "diagnosis": "Malaria",
        "nights": 3,
        "total_cost": 6000.0,
        "country": "Kenya",
        "currency": "KES",
        "policy_number": "TUR-KE-HC-001-88234",
    }
    decision = validator.validate(data, confidence=0.3)
    assert decision.status == "REVIEW"
    assert decision.approved_amount == 0.0


# ── Completeness check ─────────────────────────────────────────────────────────

def test_missing_diagnosis_returns_rejected(validator):
    data = {
        "hospital_name": "Kenyatta National Hospital",
        "diagnosis": None,
        "nights": 3,
        "total_cost": 6000.0,
        "country": "Kenya",
        "currency": "KES",
    }
    decision = validator.validate(data, confidence=0.9)
    assert decision.status == "REJECTED"
    assert "diagnosis" in decision.reason.lower() or "critical" in decision.reason.lower()


def test_missing_nights_returns_rejected(validator):
    data = {
        "hospital_name": "Kenyatta National Hospital",
        "diagnosis": "Malaria",
        "nights": None,
        "total_cost": 6000.0,
        "country": "Kenya",
        "currency": "KES",
    }
    decision = validator.validate(data, confidence=0.9)
    assert decision.status == "REJECTED"


# ── Hospital check ─────────────────────────────────────────────────────────────

def test_untrusted_hospital_returns_rejected(validator):
    data = {
        "hospital_name": "Shady Backstreet Clinic",
        "diagnosis": "Malaria",
        "nights": 3,
        "total_cost": 6000.0,
        "country": "Kenya",
        "currency": "KES",
    }
    decision = validator.validate(data, confidence=0.9)
    assert decision.status == "REJECTED"
    assert "trusted" in decision.reason.lower() or "network" in decision.reason.lower()


# ── Coverage check ─────────────────────────────────────────────────────────────

def test_excluded_diagnosis_returns_rejected(validator):
    data = {
        "hospital_name": "Kenyatta National Hospital",
        "diagnosis": "Cosmetic rhinoplasty",
        "nights": 2,
        "total_cost": 5000.0,
        "country": "Kenya",
        "currency": "KES",
    }
    decision = validator.validate(data, confidence=0.9)
    assert decision.status == "REJECTED"
    assert "exclusion" in decision.reason.lower() or "cosmetic" in decision.reason.lower()


# ── Limits check ───────────────────────────────────────────────────────────────

def test_over_limit_nights_returns_rejected(validator):
    data = {
        "hospital_name": "Nairobi Hospital",
        "diagnosis": "Pneumonia",
        "nights": 15,
        "total_cost": 40000.0,
        "country": "Kenya",
        "currency": "KES",
    }
    decision = validator.validate(data, confidence=0.9)
    assert decision.status == "REJECTED"
    assert "15" in decision.reason or "exceed" in decision.reason.lower()


# ── Approved path ──────────────────────────────────────────────────────────────

def test_valid_kenya_claim_approved(validator):
    data = {
        "hospital_name": "Kenyatta National Hospital",
        "diagnosis": "Malaria",
        "diagnosis_code": "B54",
        "nights": 3,
        "total_cost": 6000.0,
        "country": "Kenya",
        "currency": "KES",
        "policy_number": "TUR-KE-HC-001-88234",
    }
    decision = validator.validate(data, confidence=0.9)
    assert decision.status == "APPROVED"
    assert decision.approved_amount == 3000.0   # 3 nights x KES 1,000
    assert decision.currency == "KES"


def test_valid_uganda_claim_approved(validator):
    data = {
        "hospital_name": "Mulago National Referral Hospital",
        "diagnosis": "Malaria",
        "diagnosis_code": "B54",
        "nights": 2,
        "total_cost": 60000.0,
        "country": "Uganda",
        "currency": "UGX",
    }
    decision = validator.validate(data, confidence=0.9)
    assert decision.status == "APPROVED"
    assert decision.approved_amount == 60000.0  # 2 nights x UGX 30,000


def test_valid_nigeria_claim_approved(validator):
    data = {
        "hospital_name": "Lagos University Teaching Hospital",
        "diagnosis": "Typhoid Fever",
        "diagnosis_code": "A01.0",
        "nights": 4,
        "total_cost": 20000.0,
        "country": "Nigeria",
        "currency": "NGN",
    }
    decision = validator.validate(data, confidence=0.9)
    assert decision.status == "APPROVED"
    assert decision.approved_amount == 20000.0  # 4 nights x NGN 5,000


# ── Output structure ───────────────────────────────────────────────────────────

def test_decision_to_dict_has_required_keys(validator):
    data = {
        "hospital_name": "Kenyatta National Hospital",
        "diagnosis": "Malaria",
        "nights": 3,
        "total_cost": 6000.0,
        "country": "Kenya",
        "currency": "KES",
    }
    result = validator.validate(data, confidence=0.9).to_dict()
    for key in ["status", "approved_amount", "currency", "reason", "policy_matched", "checks"]:
        assert key in result, f"Missing key in decision output: {key}"


def test_checks_audit_trail_present(validator):
    data = {
        "hospital_name": "Kenyatta National Hospital",
        "diagnosis": "Malaria",
        "nights": 3,
        "total_cost": 6000.0,
        "country": "Kenya",
        "currency": "KES",
    }
    result = validator.validate(data, confidence=0.9)
    assert len(result.checks) > 0, "Audit trail must contain at least one check"
    for check in result.checks:
        assert hasattr(check, "name")
        assert hasattr(check, "passed")
        assert hasattr(check, "detail")
