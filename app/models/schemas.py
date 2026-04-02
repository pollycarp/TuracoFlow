"""
Pydantic request/response schemas for the TuracoFlow API.
"""

from typing import Optional
from pydantic import BaseModel


# ── Request ────────────────────────────────────────────────────────────────────

class ClaimSubmitResponse(BaseModel):
    claim_id: str
    status: str                        # APPROVED | REJECTED | REVIEW | DUPLICATE_CLAIM
    approved_amount: float
    currency: str
    reason: str
    policy_matched: Optional[str]
    confidence: float
    extraction_method: str             # llava | easyocr+llm | failed
    fraud_check: str                   # clean | duplicate_image | duplicate_content
    checks: dict                       # per-step audit trail


class ClaimStatusResponse(BaseModel):
    claim_id: str
    status: str
    hospital_name: Optional[str]
    submission_date: str


class HealthResponse(BaseModel):
    status: str
    service: str
    index_ready: bool
