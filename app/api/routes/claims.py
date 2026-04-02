"""
Claims API routes.

POST /claims/submit  — accept a receipt image and return a claim decision
GET  /claims/{id}    — look up a previously submitted claim by ID
"""

import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import ClaimStatusResponse, ClaimSubmitResponse
from app.modules.extractor import extract_receipt
from app.modules.fraud import FraudDetector
from app.modules.pii_masker import mask_receipt_result
from app.modules.validator import ClaimsValidator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/claims", tags=["claims"])

# Module-level singletons — initialised once, reused across requests
_validator = ClaimsValidator()
_fraud = FraudDetector()


@router.post("/submit", response_model=ClaimSubmitResponse)
async def submit_claim(
    image: UploadFile = File(..., description="Receipt image (JPEG or PNG)"),
    customer_id: str = Form(..., description="Customer identifier"),
    policy_id: str = Form(..., description="Turaco policy number"),
):
    """
    Full claims pipeline:
      1. Save uploaded image to a temp file
      2. Extract structured data from receipt (LLaVA → EasyOCR fallback)
      3. Mask PII before any further processing
      4. Fraud check (duplicate image or content hash)
      5. Validate against policy knowledge base
      6. Record result and return decision
    """
    claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
    logger.info(f"Processing claim {claim_id} for customer {customer_id}")

    # ── Step 1: Save image temporarily ────────────────────────────────────────
    suffix = Path(image.filename).suffix if image.filename else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        # ── Step 2: Extract ───────────────────────────────────────────────────
        extraction = extract_receipt(tmp_path)
        confidence = extraction["confidence"]
        method = extraction["method"]

        if method == "failed":
            raise HTTPException(
                status_code=422,
                detail="Could not extract data from the receipt image. Please upload a clearer photo.",
            )

        # ── Step 3: Mask PII ──────────────────────────────────────────────────
        masked = mask_receipt_result(extraction)
        extracted_data = masked["data"]

        # Inject policy_id from form if not found in image
        if not extracted_data.get("policy_number"):
            extracted_data["policy_number"] = policy_id

        # ── Step 4: Fraud check ───────────────────────────────────────────────
        fraud_result = _fraud.check(claim_id, tmp_path, extracted_data)

        if fraud_result["is_duplicate"]:
            dup_type = fraud_result["duplicate_type"]
            matched = fraud_result["matched_claim_id"]
            logger.warning(f"Duplicate claim {claim_id} matches {matched} ({dup_type})")
            # Record the duplicate so it can be looked up by claim_id
            _fraud.record(
                claim_id=claim_id,
                image_hash=fraud_result["image_hash"],
                content_hash=fraud_result["content_hash"],
                extracted_data=extracted_data,
                status="DUPLICATE_CLAIM",
            )
            return ClaimSubmitResponse(
                claim_id=claim_id,
                status="DUPLICATE_CLAIM",
                approved_amount=0.0,
                currency=extracted_data.get("currency") or "",
                reason=f"This receipt has already been submitted (matched claim: {matched}).",
                policy_matched=None,
                confidence=confidence,
                extraction_method=method,
                fraud_check=f"duplicate_{dup_type}",
                checks={},
            )

        # ── Step 5: Validate ──────────────────────────────────────────────────
        decision = _validator.validate(extracted_data, confidence)

        # ── Step 6: Record ────────────────────────────────────────────────────
        _fraud.record(
            claim_id=claim_id,
            image_hash=fraud_result["image_hash"],
            content_hash=fraud_result["content_hash"],
            extracted_data=extracted_data,
            status=decision.status,
        )

        return ClaimSubmitResponse(
            claim_id=claim_id,
            status=decision.status,
            approved_amount=decision.approved_amount,
            currency=decision.currency,
            reason=decision.reason,
            policy_matched=decision.policy_matched,
            confidence=confidence,
            extraction_method=method,
            fraud_check="clean",
            checks=decision.to_dict()["checks"],
        )

    finally:
        # Always clean up the temp file
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/{claim_id}", response_model=ClaimStatusResponse)
def get_claim(claim_id: str):
    """Look up a previously submitted claim by its ID."""
    record = _fraud.get_claim(claim_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")
    return ClaimStatusResponse(
        claim_id=record["claim_id"],
        status=record["status"],
        hospital_name=record.get("hospital_name"),
        submission_date=record["submission_date"],
    )
