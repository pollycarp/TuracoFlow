"""
RAG Pipeline — Policy Knowledge Base

Reads policy text files, chunks them by section, embeds using
sentence-transformers (local, free), and stores in LanceDB.

Retrieval supports optional filtering by country and product_type
so Kenya queries never bleed into Nigeria results.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

# Embedding model — 80 MB, runs fully local, no API needed
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# Sections we want to keep as individual chunks
SECTION_HEADERS = [
    "OVERVIEW",
    "ELIGIBILITY",
    "COVERAGE DETAILS",
    "COVERED CONDITIONS",
    "EXCLUSIONS",
    "CLAIM REQUIREMENTS",
    "PAYOUT",
    "CLAIM LIMITS SUMMARY",
    "DISABILITY SCALE",
    "COVERED CAUSES OF DEATH",
    "COVERED ACCIDENT TYPES",
]


def _parse_policy_metadata(text: str, filename: str) -> dict:
    """Extract country, product_type, policy_code from the document header."""
    country = "Unknown"
    product_type = "Unknown"
    policy_code = "Unknown"

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Country:"):
            country = line.split(":", 1)[1].strip()
        elif line.startswith("Product Type:"):
            product_type = line.split(":", 1)[1].strip()
        elif line.startswith("Policy Code:"):
            policy_code = line.split(":", 1)[1].strip()

    return {"country": country, "product_type": product_type, "policy_code": policy_code}


def _chunk_policy(text: str, metadata: dict) -> list[dict]:
    """
    Split a policy document into section-level chunks.
    Each chunk carries full metadata so it can be filtered independently.
    """
    chunks = []
    current_section = "HEADER"
    current_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        # Check if this line is a known section header
        is_header = any(
            stripped == header or stripped.startswith(header + " ")
            for header in SECTION_HEADERS
        )

        if is_header and stripped:
            # Save the previous section
            if current_lines:
                content = "\n".join(current_lines).strip()
                if len(content) > 30:  # skip near-empty sections
                    chunks.append({
                        "section": current_section,
                        "content": content,
                        **metadata,
                    })
            current_section = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Save the last section
    if current_lines:
        content = "\n".join(current_lines).strip()
        if len(content) > 30:
            chunks.append({
                "section": current_section,
                "content": content,
                **metadata,
            })

    return chunks


class PolicyRAG:
    """
    Manages the policy knowledge base:
    - build_index(): reads all policy files, embeds, stores in LanceDB
    - retrieve(): vector search with optional country/product_type filter
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.db = lancedb.connect(settings.lancedb_path)
        self._table = None

    @property
    def table(self):
        if self._table is None:
            try:
                self._table = self.db.open_table(settings.lancedb_table_name)
            except Exception:
                self._table = None
        return self._table

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    def build_index(self, policies_dir: Optional[str] = None) -> int:
        """
        Read all .txt policy files, chunk by section, embed, and store.
        Returns the number of chunks indexed.
        """
        if policies_dir is None:
            policies_dir = Path(__file__).parent.parent.parent / "data" / "policies"

        policy_files = list(Path(policies_dir).glob("*.txt"))
        if not policy_files:
            raise FileNotFoundError(f"No policy files found in {policies_dir}")

        all_chunks = []
        for filepath in policy_files:
            text = filepath.read_text(encoding="utf-8")
            metadata = _parse_policy_metadata(text, filepath.stem)
            chunks = _chunk_policy(text, metadata)
            logger.info(f"Parsed {filepath.name}: {len(chunks)} chunks ({metadata['country']})")
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No chunks extracted from policy files.")

        # Embed all chunk contents
        texts = [c["content"] for c in all_chunks]
        logger.info(f"Embedding {len(all_chunks)} chunks with {EMBED_MODEL}...")
        vectors = self._embed(texts)

        # Build records for LanceDB
        records = []
        for i, (chunk, vector) in enumerate(zip(all_chunks, vectors)):
            records.append({
                "id": f"{chunk['policy_code']}_{chunk['section'].replace(' ', '_')}_{i}",
                "policy_code": chunk["policy_code"],
                "country": chunk["country"],
                "product_type": chunk["product_type"],
                "section": chunk["section"],
                "content": chunk["content"],
                "vector": vector,
            })

        # Drop and recreate the table so re-indexing is idempotent
        try:
            self.db.drop_table(settings.lancedb_table_name)
        except Exception:
            pass

        self._table = self.db.create_table(settings.lancedb_table_name, data=records)
        logger.info(f"Indexed {len(records)} chunks into LanceDB table '{settings.lancedb_table_name}'")
        return len(records)

    def retrieve(
        self,
        query: str,
        country: Optional[str] = None,
        product_type: Optional[str] = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Semantic search over the policy knowledge base.

        Args:
            query: Natural language question, e.g. "Is malaria covered in Kenya?"
            country: Optional filter, e.g. "Kenya"
            product_type: Optional filter, e.g. "HospiCash"
            top_k: Number of chunks to return

        Returns:
            List of chunk dicts with keys: country, product_type, section, content, _distance
        """
        if self.table is None:
            raise RuntimeError("Policy index not built. Call build_index() first.")

        query_vector = self._embed([query])[0]

        search = self.table.search(query_vector)

        # Apply metadata filters to prevent cross-country contamination
        filters = []
        if country:
            filters.append(f"country = '{country}'")
        if product_type:
            filters.append(f"product_type = '{product_type}'")
        if filters:
            search = search.where(" AND ".join(filters))

        results = search.limit(top_k).to_list()

        return [
            {
                "country": r["country"],
                "product_type": r["product_type"],
                "policy_code": r["policy_code"],
                "section": r["section"],
                "content": r["content"],
                "score": round(max(0.0, 1 - r.get("_distance", 0)), 4),
            }
            for r in results
        ]

    def is_indexed(self) -> bool:
        """Return True if the policy table exists and has data."""
        try:
            tbl = self.db.open_table(settings.lancedb_table_name)
            return tbl.count_rows() > 0
        except Exception:
            return False
