"""
Fraud Detection Module (Phase 7)

Detects duplicate claims by hashing the receipt image and the extracted
data, then checking against a SQLite database of previously processed claims.

Two-layer hashing strategy:
  1. Image hash  (perceptual hash via imagehash) — catches resubmitted photos
  2. Content hash (SHA-256 of key fields)        — catches same data, different photo

A claim is flagged as DUPLICATE_CLAIM if either hash has been seen before.

SQLite is built into Python — zero extra dependencies.
"""

import hashlib
import imagehash
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

# Fields used to build the content hash — enough to uniquely identify a claim
_CONTENT_HASH_FIELDS = [
    "patient_name",
    "hospital_name",
    "admission_date",
    "discharge_date",
    "diagnosis",
    "total_cost",
    "currency",
    "policy_number",
]


# ── Database setup ─────────────────────────────────────────────────────────────

def _get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Create the claims table if it does not exist."""
    db_path = db_path or settings.sqlite_db_path
    with _get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_claims (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id        TEXT NOT NULL,
                image_hash      TEXT NOT NULL,
                content_hash    TEXT NOT NULL,
                policy_number   TEXT,
                hospital_name   TEXT,
                submission_date TEXT NOT NULL,
                status          TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_image_hash   ON processed_claims(image_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON processed_claims(content_hash)")
        conn.commit()
    logger.info(f"Fraud DB initialised at {db_path}")


# ── Hashing ────────────────────────────────────────────────────────────────────

def compute_image_hash(image_path: str) -> str:
    """
    Perceptual hash of the receipt image.
    Robust to minor compression artefacts and resizing.
    Two images are considered the same if their hash distance ≤ 8.
    """
    img = Image.open(image_path)
    return str(imagehash.phash(img))


def compute_content_hash(extracted_data: dict) -> str:
    """
    SHA-256 of the key claim fields.
    Catches duplicate submissions with the same data but a different photo.
    """
    payload = {k: extracted_data.get(k) for k in _CONTENT_HASH_FIELDS}
    serialised = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


def _image_hashes_similar(hash_a: str, hash_b: str, threshold: int = 8) -> bool:
    """Return True if two perceptual hashes are within the similarity threshold."""
    try:
        return (imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)) <= threshold
    except Exception:
        return hash_a == hash_b


# ── Fraud checker ──────────────────────────────────────────────────────────────

class FraudDetector:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.sqlite_db_path
        init_db(self.db_path)

    def check(
        self,
        claim_id: str,
        image_path: str,
        extracted_data: dict,
    ) -> dict:
        """
        Check whether this claim is a duplicate of a previously processed one.

        Returns a dict:
          {
            "is_duplicate": bool,
            "duplicate_type": "image" | "content" | None,
            "matched_claim_id": str | None,
            "image_hash": str,
            "content_hash": str,
          }
        """
        image_hash = compute_image_hash(image_path)
        content_hash = compute_content_hash(extracted_data)

        with _get_connection(self.db_path) as conn:
            # Check content hash (exact duplicate data)
            row = conn.execute(
                "SELECT claim_id FROM processed_claims WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()

            if row:
                logger.warning(f"Duplicate content hash detected. Original claim: {row['claim_id']}")
                return {
                    "is_duplicate": True,
                    "duplicate_type": "content",
                    "matched_claim_id": row["claim_id"],
                    "image_hash": image_hash,
                    "content_hash": content_hash,
                }

            # Check image hash (same photo, possibly re-edited)
            rows = conn.execute(
                "SELECT claim_id, image_hash FROM processed_claims"
            ).fetchall()

            for existing in rows:
                if _image_hashes_similar(image_hash, existing["image_hash"]):
                    logger.warning(f"Duplicate image hash detected. Original claim: {existing['claim_id']}")
                    return {
                        "is_duplicate": True,
                        "duplicate_type": "image",
                        "matched_claim_id": existing["claim_id"],
                        "image_hash": image_hash,
                        "content_hash": content_hash,
                    }

        return {
            "is_duplicate": False,
            "duplicate_type": None,
            "matched_claim_id": None,
            "image_hash": image_hash,
            "content_hash": content_hash,
        }

    def record(
        self,
        claim_id: str,
        image_hash: str,
        content_hash: str,
        extracted_data: dict,
        status: str,
    ) -> None:
        """
        Persist a processed claim's hashes to the database.
        Call this AFTER a claim has been validated and a decision made.
        """
        with _get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO processed_claims
                    (claim_id, image_hash, content_hash, policy_number,
                     hospital_name, submission_date, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim_id,
                    image_hash,
                    content_hash,
                    extracted_data.get("policy_number"),
                    extracted_data.get("hospital_name"),
                    datetime.now(timezone.utc).isoformat(),
                    status,
                ),
            )
            conn.commit()
        logger.info(f"Recorded claim {claim_id} with status {status}")

    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Retrieve a stored claim record by claim_id."""
        with _get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM processed_claims WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        """Return total number of processed claims in the database."""
        with _get_connection(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM processed_claims"
            ).fetchone()[0]

    def clear(self) -> None:
        """Wipe all records — used in tests only."""
        with _get_connection(self.db_path) as conn:
            conn.execute("DELETE FROM processed_claims")
            conn.commit()
