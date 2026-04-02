"""
Unit tests for the Fraud Detection module (Phase 7).

Tests verify:
1. Clean claim passes (not a duplicate)
2. Same receipt image resubmitted → flagged as image duplicate
3. Same extracted data, different claim ID → flagged as content duplicate
4. Different claim, different data → not flagged
5. Recording a claim persists it to SQLite
6. claim_id is retrievable after recording
7. Count increments correctly
8. Image hash is stable (same image → same hash)
9. Content hash is stable (same data → same hash)
10. Slightly different data produces different content hash
"""

import pytest
from pathlib import Path

from app.modules.fraud import (
    FraudDetector,
    compute_image_hash,
    compute_content_hash,
    _image_hashes_similar,
)

RECEIPTS_DIR = Path(__file__).parent.parent / "data" / "receipts"
TEST_DB = "./test_fraud.db"


@pytest.fixture(autouse=True)
def clean_db():
    """Wipe all records before and after each test.
    We clear rather than delete the file — Windows locks SQLite files
    while connections are open, making os.remove() unreliable."""
    detector = FraudDetector(db_path=TEST_DB)
    detector.clear()
    yield
    detector.clear()


@pytest.fixture
def detector():
    return FraudDetector(db_path=TEST_DB)


@pytest.fixture
def sample_data():
    return {
        "patient_name": "[REDACTED NAME]",
        "hospital_name": "Kenyatta National Hospital",
        "admission_date": "2026-03-24",
        "discharge_date": "2026-03-27",
        "diagnosis": "Malaria",
        "total_cost": 6000.0,
        "currency": "KES",
        "policy_number": "TUR-KE-HC-001-88234",
    }


@pytest.fixture
def approved_image():
    return str(RECEIPTS_DIR / "receipt_approved.png")


@pytest.fixture
def over_limit_image():
    return str(RECEIPTS_DIR / "receipt_over_limit.png")


# ── Hash stability ─────────────────────────────────────────────────────────────

def test_image_hash_is_stable(approved_image):
    hash1 = compute_image_hash(approved_image)
    hash2 = compute_image_hash(approved_image)
    assert hash1 == hash2


def test_content_hash_is_stable(sample_data):
    hash1 = compute_content_hash(sample_data)
    hash2 = compute_content_hash(sample_data)
    assert hash1 == hash2


def test_different_data_produces_different_hash(sample_data):
    hash1 = compute_content_hash(sample_data)
    modified = {**sample_data, "total_cost": 9999.0}
    hash2 = compute_content_hash(modified)
    assert hash1 != hash2


def test_same_image_hashes_are_similar(approved_image):
    h = compute_image_hash(approved_image)
    assert _image_hashes_similar(h, h)


def test_different_images_are_not_similar(approved_image, over_limit_image):
    h1 = compute_image_hash(approved_image)
    h2 = compute_image_hash(over_limit_image)
    assert not _image_hashes_similar(h1, h2)


# ── Clean claim (first submission) ────────────────────────────────────────────

def test_first_submission_is_not_duplicate(detector, approved_image, sample_data):
    result = detector.check("CLAIM-001", approved_image, sample_data)
    assert result["is_duplicate"] is False
    assert result["duplicate_type"] is None
    assert result["matched_claim_id"] is None


# ── Image duplicate ────────────────────────────────────────────────────────────

def test_same_image_resubmitted_flagged_as_duplicate(detector, approved_image, sample_data):
    # First submission
    result1 = detector.check("CLAIM-001", approved_image, sample_data)
    detector.record("CLAIM-001", result1["image_hash"], result1["content_hash"], sample_data, "APPROVED")

    # Second submission — same image
    different_data = {**sample_data, "policy_number": "TUR-KE-HC-001-99999"}
    result2 = detector.check("CLAIM-002", approved_image, different_data)

    assert result2["is_duplicate"] is True
    assert result2["duplicate_type"] == "image"
    assert result2["matched_claim_id"] == "CLAIM-001"


# ── Content duplicate ──────────────────────────────────────────────────────────

def test_same_data_different_image_flagged_as_content_duplicate(
    detector, approved_image, over_limit_image, sample_data
):
    # Record the first claim with the approved image
    result1 = detector.check("CLAIM-001", approved_image, sample_data)
    detector.record("CLAIM-001", result1["image_hash"], result1["content_hash"], sample_data, "APPROVED")

    # Submit same extracted data but with a different image
    result2 = detector.check("CLAIM-002", over_limit_image, sample_data)

    assert result2["is_duplicate"] is True
    assert result2["duplicate_type"] == "content"
    assert result2["matched_claim_id"] == "CLAIM-001"


# ── Distinct claims ────────────────────────────────────────────────────────────

def test_distinct_claims_not_flagged(detector, approved_image, over_limit_image, sample_data):
    # Record first claim
    result1 = detector.check("CLAIM-001", approved_image, sample_data)
    detector.record("CLAIM-001", result1["image_hash"], result1["content_hash"], sample_data, "APPROVED")

    # Different patient, different image
    different_data = {
        **sample_data,
        "patient_name": "[REDACTED NAME]",
        "hospital_name": "Nairobi Hospital",
        "admission_date": "2026-03-10",
        "discharge_date": "2026-03-25",
        "total_cost": 40000.0,
        "policy_number": "TUR-KE-HC-001-71092",
    }
    result2 = detector.check("CLAIM-002", over_limit_image, different_data)
    assert result2["is_duplicate"] is False


# ── Record and retrieve ────────────────────────────────────────────────────────

def test_record_persists_to_db(detector, approved_image, sample_data):
    result = detector.check("CLAIM-001", approved_image, sample_data)
    detector.record("CLAIM-001", result["image_hash"], result["content_hash"], sample_data, "APPROVED")
    assert detector.count() == 1


def test_get_claim_returns_record(detector, approved_image, sample_data):
    result = detector.check("CLAIM-001", approved_image, sample_data)
    detector.record("CLAIM-001", result["image_hash"], result["content_hash"], sample_data, "APPROVED")

    record = detector.get_claim("CLAIM-001")
    assert record is not None
    assert record["claim_id"] == "CLAIM-001"
    assert record["status"] == "APPROVED"
    assert record["hospital_name"] == "Kenyatta National Hospital"


def test_get_unknown_claim_returns_none(detector):
    assert detector.get_claim("NONEXISTENT") is None


def test_count_increments_with_each_record(detector, approved_image, over_limit_image, sample_data):
    assert detector.count() == 0

    r1 = detector.check("CLAIM-001", approved_image, sample_data)
    detector.record("CLAIM-001", r1["image_hash"], r1["content_hash"], sample_data, "APPROVED")
    assert detector.count() == 1

    other_data = {**sample_data, "total_cost": 40000.0, "policy_number": "TUR-KE-HC-001-71092"}
    r2 = detector.check("CLAIM-002", over_limit_image, other_data)
    detector.record("CLAIM-002", r2["image_hash"], r2["content_hash"], other_data, "REJECTED")
    assert detector.count() == 2
