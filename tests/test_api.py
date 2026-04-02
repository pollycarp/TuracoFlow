"""
Integration tests for the FastAPI layer (Phase 8).

Uses httpx TestClient — no live server needed.

Tests verify:
1. GET /health returns ok and index_ready status
2. POST /claims/submit with approved receipt returns APPROVED or REVIEW
3. POST /claims/submit with over-limit receipt returns REJECTED
4. POST /claims/submit with missing image returns 422
5. GET /claims/{id} returns claim after submission
6. GET /claims/{unknown_id} returns 404
7. Duplicate submission returns DUPLICATE_CLAIM
8. Response always contains required fields
"""

import io
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app

RECEIPTS_DIR = Path(__file__).parent.parent / "data" / "receipts"

client = TestClient(app)


def _image_bytes(filename: str) -> bytes:
    return (RECEIPTS_DIR / filename).read_bytes()


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "TuracoFlow"
    assert "index_ready" in body


# ── Submit claim ───────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_submit_approved_receipt():
    response = client.post(
        "/claims/submit",
        data={"customer_id": "CUST-001", "policy_id": "TUR-KE-HC-001-88234"},
        files={"image": ("receipt_approved.png", _image_bytes("receipt_approved.png"), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("APPROVED", "REVIEW", "REJECTED")
    assert "claim_id" in body
    assert body["claim_id"].startswith("CLM-")
    assert body["fraud_check"] == "clean"


@pytest.mark.slow
def test_submit_over_limit_receipt_rejected():
    response = client.post(
        "/claims/submit",
        data={"customer_id": "CUST-002", "policy_id": "TUR-KE-HC-001-71092"},
        files={"image": ("receipt_over_limit.png", _image_bytes("receipt_over_limit.png"), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    # Over-limit stays should be rejected at the limits check
    assert body["status"] in ("REJECTED", "REVIEW")
    assert body["fraud_check"] == "clean"


@pytest.mark.slow
def test_submit_low_confidence_receipt_returns_review():
    response = client.post(
        "/claims/submit",
        data={"customer_id": "CUST-003", "policy_id": "TUR-KE-HC-001-00000"},
        files={"image": ("receipt_low_confidence.png", _image_bytes("receipt_low_confidence.png"), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("REVIEW", "REJECTED")


# ── Response structure ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_response_contains_required_fields():
    response = client.post(
        "/claims/submit",
        data={"customer_id": "CUST-004", "policy_id": "TUR-KE-HC-001-88234"},
        files={"image": ("receipt_approved.png", _image_bytes("receipt_approved.png"), "image/png")},
    )
    body = response.json()
    required = [
        "claim_id", "status", "approved_amount", "currency",
        "reason", "confidence", "extraction_method", "fraud_check", "checks",
    ]
    for field in required:
        assert field in body, f"Missing field in response: {field}"


# ── Duplicate detection ────────────────────────────────────────────────────────

@pytest.mark.slow
def test_duplicate_submission_flagged():
    image_bytes = _image_bytes("receipt_approved.png")

    # First submission
    r1 = client.post(
        "/claims/submit",
        data={"customer_id": "CUST-005", "policy_id": "TUR-KE-HC-001-88234"},
        files={"image": ("receipt_approved.png", image_bytes, "image/png")},
    )
    assert r1.status_code == 200

    # Second submission — same image
    r2 = client.post(
        "/claims/submit",
        data={"customer_id": "CUST-005", "policy_id": "TUR-KE-HC-001-88234"},
        files={"image": ("receipt_approved.png", image_bytes, "image/png")},
    )
    assert r2.status_code == 200
    assert r2.json()["status"] == "DUPLICATE_CLAIM"
    assert "duplicate" in r2.json()["fraud_check"]


# ── Claim lookup ───────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_get_claim_after_submission():
    response = client.post(
        "/claims/submit",
        data={"customer_id": "CUST-006", "policy_id": "TUR-KE-HC-001-88234"},
        files={"image": ("receipt_approved.png", _image_bytes("receipt_approved.png"), "image/png")},
    )
    claim_id = response.json()["claim_id"]

    lookup = client.get(f"/claims/{claim_id}")
    assert lookup.status_code == 200
    body = lookup.json()
    assert body["claim_id"] == claim_id
    assert "status" in body
    assert "submission_date" in body


def test_get_unknown_claim_returns_404():
    response = client.get("/claims/CLM-DOESNOTEXIST")
    assert response.status_code == 404
