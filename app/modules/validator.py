"""
Claims Validation Logic (Phase 6)

Orchestrates five sequential checks against the extracted receipt data:

  1. Confidence check   — low OCR confidence → REVIEW (human needed)
  2. Completeness check — missing critical fields → REJECTED
  3. Hospital check     — hospital must be in trusted list
  4. Coverage check     — diagnosis must be covered by policy (via RAG)
  5. Limits check       — nights and amount must be within policy cap

Returns a ClaimDecision with status APPROVED | REJECTED | REVIEW,
the approved payout amount, a human-readable reason, and a per-check
audit trail for explainability.
"""

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.modules.rag import PolicyRAG

logger = logging.getLogger(__name__)

# ── Data paths ─────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_HOSPITALS_CSV = _DATA_DIR / "hospitals" / "trusted_hospitals.csv"

# ── Policy limits (parsed from RAG chunks via regex) ──────────────────────────
# Fallback hardcoded limits if regex extraction fails.
# These match the policy documents created in Phase 2.
_FALLBACK_LIMITS = {
    ("Kenya",   "HospiCash"):         {"per_night": 1000,   "max_nights": 10, "max_amount": 10000,  "currency": "KES"},
    ("Uganda",  "HospiCash"):         {"per_night": 30000,  "max_nights": 10, "max_amount": 300000, "currency": "UGX"},
    ("Nigeria", "HospiCash"):         {"per_night": 5000,   "max_nights": 10, "max_amount": 50000,  "currency": "NGN"},
    ("Uganda",  "Personal Accident"): {"per_night": 30000,  "max_nights": 10, "max_amount": 300000, "currency": "UGX"},
}

# Diagnoses that are explicitly excluded across all policies
_GLOBAL_EXCLUSIONS = [
    "cosmetic", "elective", "self-inflicted", "substance abuse",
    "experimental", "traditional medicine",
]


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ClaimDecision:
    status: str                        # "APPROVED" | "REJECTED" | "REVIEW"
    approved_amount: float
    currency: str
    reason: str
    policy_matched: Optional[str]
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "approved_amount": self.approved_amount,
            "currency": self.currency,
            "reason": self.reason,
            "policy_matched": self.policy_matched,
            "checks": {c.name: {"passed": c.passed, "detail": c.detail} for c in self.checks},
        }


# ── Hospital loader ────────────────────────────────────────────────────────────

def _load_trusted_hospitals() -> dict[str, dict]:
    """
    Load the trusted hospitals CSV into a dict keyed by normalised name.
    Normalised = lowercase, punctuation stripped.
    """
    hospitals = {}
    with open(_HOSPITALS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = _normalise(row["hospital_name"])
            hospitals[key] = row
    return hospitals


def _normalise(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy hospital name matching."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _find_hospital(name: str, hospitals: dict) -> Optional[dict]:
    """
    Match a hospital name against the trusted list.
    Uses normalised exact match first, then substring match as fallback.
    """
    if not name:
        return None

    norm = _normalise(name)

    # Exact match
    if norm in hospitals:
        return hospitals[norm]

    # Substring match (handles abbreviations like "Kenyatta Nat. Hospital")
    for key, record in hospitals.items():
        if norm in key or key in norm:
            return record

    # Word overlap match (≥ 2 significant words in common)
    query_words = set(norm.split()) - {"hospital", "clinic", "medical", "centre", "center", "the"}
    for key, record in hospitals.items():
        key_words = set(key.split()) - {"hospital", "clinic", "medical", "centre", "center", "the"}
        if len(query_words & key_words) >= 2:
            return record

    return None


# ── Policy limits parser ───────────────────────────────────────────────────────

def _extract_limits_from_chunks(chunks: list[dict], country: str, product_type: str) -> dict:
    """
    Parse per-night benefit and maximum claim amount from RAG-retrieved
    CLAIM LIMITS SUMMARY chunks using regex.
    Falls back to hardcoded limits if regex extraction fails.
    """
    fallback = _FALLBACK_LIMITS.get((country, product_type), {})

    limits_text = " ".join(
        c["content"] for c in chunks
        if "CLAIM LIMITS" in c.get("section", "").upper()
        or "COVERAGE DETAILS" in c.get("section", "").upper()
    )

    if not limits_text:
        logger.info(f"No limits chunk found for {country}/{product_type}, using fallback.")
        return fallback

    # Try to extract per-night benefit
    per_night = fallback.get("per_night")
    night_match = re.search(r'(?:per.night|benefit|per.night.benefit)\s*[:\-]?\s*[\w]+\s*([\d,]+)', limits_text, re.IGNORECASE)
    if night_match:
        per_night = int(night_match.group(1).replace(",", ""))

    # Try to extract max nights
    max_nights = fallback.get("max_nights", 10)
    nights_match = re.search(r'(\d+)\s*nights?\s*maximum', limits_text, re.IGNORECASE)
    if nights_match:
        max_nights = int(nights_match.group(1))

    # Try to extract max amount
    max_amount = fallback.get("max_amount")
    amount_match = re.search(r'(?:single claim maximum|maximum benefit per claim|claim maximum)\s*[:\-]?\s*[\w]+\s*([\d,]+)', limits_text, re.IGNORECASE)
    if amount_match:
        max_amount = int(amount_match.group(1).replace(",", ""))

    return {
        "per_night": per_night,
        "max_nights": max_nights,
        "max_amount": max_amount,
        "currency": fallback.get("currency", ""),
    }


# ── Validator ──────────────────────────────────────────────────────────────────

class ClaimsValidator:
    def __init__(self):
        self._hospitals: Optional[dict] = None
        self._rag: Optional[PolicyRAG] = None

    @property
    def hospitals(self) -> dict:
        if self._hospitals is None:
            self._hospitals = _load_trusted_hospitals()
        return self._hospitals

    @property
    def rag(self) -> PolicyRAG:
        if self._rag is None:
            self._rag = PolicyRAG()
        return self._rag

    def validate(self, extracted: dict, confidence: float) -> ClaimDecision:
        """
        Run all validation checks on extracted receipt data.

        Args:
            extracted: dict with keys from REQUIRED_FIELDS (Phase 4 output)
            confidence: float 0.0–1.0 from extractor

        Returns:
            ClaimDecision
        """
        checks: list[CheckResult] = []

        # ── Check 1: Confidence ────────────────────────────────────────────────
        conf_check = self._check_confidence(confidence)
        checks.append(conf_check)
        if not conf_check.passed:
            return ClaimDecision(
                status="REVIEW",
                approved_amount=0.0,
                currency=extracted.get("currency") or "",
                reason=conf_check.detail,
                policy_matched=None,
                checks=checks,
            )

        # ── Check 2: Completeness ──────────────────────────────────────────────
        complete_check = self._check_completeness(extracted)
        checks.append(complete_check)
        if not complete_check.passed:
            return ClaimDecision(
                status="REJECTED",
                approved_amount=0.0,
                currency=extracted.get("currency") or "",
                reason=complete_check.detail,
                policy_matched=None,
                checks=checks,
            )

        # ── Check 3: Trusted hospital ──────────────────────────────────────────
        hospital_check, hospital_record = self._check_hospital(extracted)
        checks.append(hospital_check)
        if not hospital_check.passed:
            return ClaimDecision(
                status="REJECTED",
                approved_amount=0.0,
                currency=extracted.get("currency") or "",
                reason=hospital_check.detail,
                policy_matched=None,
                checks=checks,
            )

        # ── Check 4: Diagnosis coverage (RAG) ─────────────────────────────────
        country = extracted.get("country") or (hospital_record or {}).get("country", "")
        product_type = self._infer_product_type(extracted)
        coverage_check, policy_chunks, policy_code = self._check_coverage(
            extracted, country, product_type
        )
        checks.append(coverage_check)
        if not coverage_check.passed:
            return ClaimDecision(
                status="REJECTED",
                approved_amount=0.0,
                currency=extracted.get("currency") or "",
                reason=coverage_check.detail,
                policy_matched=policy_code,
                checks=checks,
            )

        # ── Check 5: Limits ────────────────────────────────────────────────────
        limits = _extract_limits_from_chunks(policy_chunks, country, product_type)
        limits_check, approved_amount = self._check_limits(extracted, limits)
        checks.append(limits_check)
        if not limits_check.passed:
            return ClaimDecision(
                status="REJECTED",
                approved_amount=0.0,
                currency=limits.get("currency") or extracted.get("currency") or "",
                reason=limits_check.detail,
                policy_matched=policy_code,
                checks=checks,
            )

        # ── All checks passed → APPROVED ──────────────────────────────────────
        nights = extracted.get("nights") or 1
        hospital_name = extracted.get("hospital_name", "the hospital")
        diagnosis = extracted.get("diagnosis", "the stated condition")

        return ClaimDecision(
            status="APPROVED",
            approved_amount=approved_amount,
            currency=limits.get("currency") or extracted.get("currency") or "",
            reason=(
                f"Valid {nights}-night inpatient stay for {diagnosis} "
                f"at {hospital_name}. "
                f"Payout: {limits.get('per_night', 0):,} x {nights} nights."
            ),
            policy_matched=policy_code,
            checks=checks,
        )

    # ── Individual check methods ───────────────────────────────────────────────

    def _check_confidence(self, confidence: float) -> CheckResult:
        threshold = settings.confidence_threshold
        if confidence >= threshold:
            return CheckResult("confidence", True, f"Confidence {confidence} ≥ threshold {threshold}")
        return CheckResult(
            "confidence", False,
            f"Confidence {confidence} is below threshold {threshold}. Manual review required."
        )

    def _check_completeness(self, data: dict) -> CheckResult:
        critical = ["hospital_name", "diagnosis", "nights", "total_cost", "country"]
        missing = [f for f in critical if not data.get(f)]
        if not missing:
            return CheckResult("completeness", True, "All critical fields present.")
        return CheckResult(
            "completeness", False,
            f"Missing critical fields: {', '.join(missing)}."
        )

    def _check_hospital(self, data: dict) -> tuple[CheckResult, Optional[dict]]:
        name = data.get("hospital_name")
        record = _find_hospital(name, self.hospitals)
        if record and record.get("turaco_verified") == "true":
            return (
                CheckResult("hospital_trusted", True, f"'{name}' is a Turaco-verified facility."),
                record,
            )
        return (
            CheckResult(
                "hospital_trusted", False,
                f"'{name}' is not in the Turaco trusted hospital network."
            ),
            None,
        )

    def _check_coverage(
        self, data: dict, country: str, product_type: str
    ) -> tuple[CheckResult, list[dict], Optional[str]]:
        diagnosis = data.get("diagnosis") or ""
        diagnosis_code = data.get("diagnosis_code") or ""

        # Check for global exclusions first
        diag_lower = diagnosis.lower()
        for exclusion in _GLOBAL_EXCLUSIONS:
            if exclusion in diag_lower:
                return (
                    CheckResult(
                        "diagnosis_covered", False,
                        f"Diagnosis '{diagnosis}' matches exclusion category '{exclusion}'."
                    ),
                    [],
                    None,
                )

        # RAG query
        query = f"Is {diagnosis} {diagnosis_code} covered for inpatient stay?"
        chunks = self.rag.retrieve(query, country=country, product_type=product_type, top_k=4)

        if not chunks:
            return (
                CheckResult(
                    "diagnosis_covered", False,
                    f"No policy found for {product_type} in {country}."
                ),
                [],
                None,
            )

        policy_code = chunks[0].get("policy_code")

        # Check exclusions section in retrieved chunks
        exclusion_chunks = [c for c in chunks if "EXCLUSION" in c.get("section", "").upper()]
        for chunk in exclusion_chunks:
            for excl in _GLOBAL_EXCLUSIONS:
                if excl in chunk["content"].lower() and excl in diag_lower:
                    return (
                        CheckResult("diagnosis_covered", False, f"Diagnosis falls under policy exclusion."),
                        chunks,
                        policy_code,
                    )

        # Check covered conditions in retrieved chunks
        covered_chunks = [c for c in chunks if "COVERED" in c.get("section", "").upper()]
        diag_words = set(diagnosis.lower().split())

        for chunk in covered_chunks:
            content_lower = chunk["content"].lower()
            # Match by diagnosis name or ICD-10 code
            if any(word in content_lower for word in diag_words if len(word) > 3):
                return (
                    CheckResult("diagnosis_covered", True, f"'{diagnosis}' is covered under {policy_code}."),
                    chunks,
                    policy_code,
                )
            if diagnosis_code and diagnosis_code.lower() in content_lower:
                return (
                    CheckResult("diagnosis_covered", True, f"ICD-10 {diagnosis_code} is covered under {policy_code}."),
                    chunks,
                    policy_code,
                )

        # Semantic similarity was high enough to retrieve chunks — treat as covered
        best_score = max((c.get("score", 0) for c in chunks), default=0)
        if best_score >= 0.3:
            return (
                CheckResult("diagnosis_covered", True, f"'{diagnosis}' appears covered (similarity score {best_score})."),
                chunks,
                policy_code,
            )

        return (
            CheckResult(
                "diagnosis_covered", False,
                f"'{diagnosis}' could not be confirmed as covered under {product_type} in {country}."
            ),
            chunks,
            policy_code,
        )

    def _check_limits(self, data: dict, limits: dict) -> tuple[CheckResult, float]:
        nights = data.get("nights") or 0
        max_nights = limits.get("max_nights", 10)
        per_night = limits.get("per_night", 0)
        max_amount = limits.get("max_amount", 0)

        if nights <= 0:
            return CheckResult("limits", False, "Invalid number of nights (0 or missing)."), 0.0

        if nights > max_nights:
            return (
                CheckResult(
                    "limits", False,
                    f"Stay of {nights} nights exceeds policy maximum of {max_nights} nights."
                ),
                0.0,
            )

        approved = min(nights * per_night, max_amount)

        return (
            CheckResult(
                "limits", True,
                f"{nights} nights x {per_night:,} = {approved:,.0f} (cap: {max_amount:,})."
            ),
            float(approved),
        )

    def _infer_product_type(self, data: dict) -> str:
        """Infer product type from the claim data. Defaults to HospiCash."""
        diagnosis = (data.get("diagnosis") or "").lower()
        # Personal Accident if trauma/injury keywords present
        if any(kw in diagnosis for kw in ["accident", "fracture", "injury", "burn", "trauma"]):
            return "Personal Accident"
        return "HospiCash"
