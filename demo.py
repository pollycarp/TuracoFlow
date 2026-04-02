"""
TuracoFlow — end-to-end demo script (Phase 10)

Runs the full claims pipeline on a sample receipt and prints a
human-readable decision report.  No server required.

Usage:
    python demo.py                                    # default receipt
    python demo.py data/receipts/receipt_approved.png # custom path
"""

import sys
import textwrap
import uuid
from pathlib import Path

# ── Helpers ────────────────────────────────────────────────────────────────────

WIDTH = 58

def _bar(char="─"):
    return char * WIDTH

def _header(title: str):
    pad = (WIDTH - len(title) - 2) // 2
    print(f"{'━' * pad} {title} {'━' * (WIDTH - pad - len(title) - 2)}")

def _step(n: int, total: int, label: str, detail: str = ""):
    tick = "✓"
    detail_str = f"  ({detail})" if detail else ""
    line = f"  [{n}/{total}] {label:<32}{tick}{detail_str}"
    print(line)

def _field(label: str, value, indent: int = 4):
    label_str = f"{label}:"
    print(f"{' ' * indent}{label_str:<22}{value}")


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_demo(image_path: str):
    STEPS = 5
    claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"

    print()
    _header("TuracoFlow — Claims Pipeline Demo")
    print(f"  Receipt : {image_path}")
    print(f"  Claim ID: {claim_id}")
    print(_bar())
    print()

    # ── Step 1: Policy index ──────────────────────────────────────────────────
    print("  Loading modules...")
    from app.modules.rag import PolicyRAG
    rag = PolicyRAG()
    index_status = "loaded" if rag.is_indexed() else "not found — run scripts/build_index.py"
    _step(1, STEPS, "Loading policy index...", index_status)

    # ── Step 2: Extraction ────────────────────────────────────────────────────
    from app.modules.extractor import extract_receipt
    extraction = extract_receipt(image_path)
    confidence = extraction["confidence"]
    method     = extraction["method"]

    if method == "failed":
        print()
        print("  ✗  Could not extract data from receipt.")
        print("     Please use a clearer image.")
        print()
        return

    _step(2, STEPS, "Extracting receipt data...",
          f"method: {method}, confidence: {confidence:.2f}")

    # ── Step 3: PII masking ───────────────────────────────────────────────────
    from app.modules.pii_masker import mask_receipt_result
    masked        = mask_receipt_result(extraction)
    extracted_data = masked["data"]
    redacted_count = masked.get("redacted_fields", 0)
    _step(3, STEPS, "Masking PII...",
          f"{redacted_count} field(s) redacted" if redacted_count else "no PII found")

    # ── Step 4: Fraud check ───────────────────────────────────────────────────
    from app.modules.fraud import FraudDetector
    fraud    = FraudDetector()
    fraud_result = fraud.check(claim_id, image_path, extracted_data)

    if fraud_result["is_duplicate"]:
        dup_type = fraud_result["duplicate_type"]
        matched  = fraud_result["matched_claim_id"]
        _step(4, STEPS, "Checking for fraud...",
              f"DUPLICATE {dup_type} — matches {matched}")
        print()
        print(_bar("━"))
        _field("  STATUS", "DUPLICATE_CLAIM", indent=2)
        _field("  REASON", f"Receipt already submitted (matched: {matched})", indent=2)
        print(_bar("━"))
        print()
        return

    _step(4, STEPS, "Checking for fraud...", "clean")

    # ── Step 5: Validation ────────────────────────────────────────────────────
    from app.modules.validator import ClaimsValidator
    validator = ClaimsValidator()
    decision  = validator.validate(extracted_data, confidence)
    _step(5, STEPS, "Validating claim...", decision.status)

    # Record so the claim is retrievable afterwards
    fraud.record(
        claim_id=claim_id,
        image_hash=fraud_result["image_hash"],
        content_hash=fraud_result["content_hash"],
        extracted_data=extracted_data,
        status=decision.status,
    )

    # ── Decision report ───────────────────────────────────────────────────────
    print()
    print(_bar("━"))

    status_label = {
        "APPROVED": "✓  APPROVED",
        "REJECTED": "✗  REJECTED",
        "REVIEW":   "~  NEEDS REVIEW",
    }.get(decision.status, decision.status)

    _field("  STATUS",  status_label, indent=2)

    if decision.approved_amount and decision.approved_amount > 0:
        _field("  AMOUNT",
               f"{decision.currency} {decision.approved_amount:,.0f}", indent=2)

    if decision.policy_matched:
        _field("  POLICY",  decision.policy_matched, indent=2)

    # Wrap long reason lines
    reason_lines = textwrap.wrap(decision.reason, WIDTH - 26)
    _field("  REASON",  reason_lines[0], indent=2)
    for line in reason_lines[1:]:
        print(f"{'':28}{line}")

    _field("  CONFIDENCE", f"{confidence:.0%}", indent=2)

    print(_bar("━"))

    # Per-check audit trail
    checks = decision.to_dict().get("checks", {})
    if checks:
        print()
        print("  Audit trail:")
        for check_name, result in checks.items():
            status_icon = "✓" if result.get("passed") else "✗"
            note = result.get("note", "")
            short_note = (note[:40] + "…") if len(note) > 40 else note
            print(f"    {status_icon}  {check_name:<22} {short_note}")

    print()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    default_receipt = "data/receipts/receipt_approved.png"
    image_path = sys.argv[1] if len(sys.argv) > 1 else default_receipt

    if not Path(image_path).exists():
        print(f"Error: receipt not found at '{image_path}'")
        sys.exit(1)

    run_demo(image_path)
