"""
RAG Pipeline — Policy Knowledge Base

Reads policy text files, chunks them by section, embeds using
sentence-transformers (local, free), and stores in a numpy-based
vector store (no compiled DLL dependencies).

Retrieval supports optional filtering by country and product_type
so Kenya queries never bleed into Nigeria results.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.modules.vector_store import NumpyVectorStore

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"

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
    chunks = []
    current_section = "HEADER"
    current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        is_header = any(
            stripped == header or stripped.startswith(header + " ")
            for header in SECTION_HEADERS
        )

        if is_header and stripped:
            if current_lines:
                content = "\n".join(current_lines).strip()
                if len(content) > 30:
                    chunks.append({
                        "section": current_section,
                        "content": content,
                        **metadata,
                    })
            current_section = stripped
            current_lines = []
        else:
            current_lines.append(line)

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
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.store = NumpyVectorStore(settings.lancedb_path)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    def build_index(self, policies_dir: Optional[str] = None) -> int:
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

        texts = [c["content"] for c in all_chunks]
        logger.info(f"Embedding {len(all_chunks)} chunks with {EMBED_MODEL}...")
        vectors = self._embed(texts)

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

        # Clear and rebuild (idempotent)
        self.store.clear()
        self.store.add(records)
        self.store.save()

        logger.info(f"Indexed {self.store.count()} chunks into NumpyVectorStore at '{settings.lancedb_path}'")
        return self.store.count()

    def retrieve(
        self,
        query: str,
        country: Optional[str] = None,
        product_type: Optional[str] = None,
        top_k: int = 3,
    ) -> list[dict]:
        if not self.store.exists():
            raise RuntimeError("Policy index not built. Call build_index() first.")

        if self.store.count() == 0:
            self.store.load()

        query_vector = self._embed([query])[0]

        filters = {}
        if country:
            filters["country"] = country
        if product_type:
            filters["product_type"] = product_type

        return self.store.search(query_vector, filters=filters or None, top_k=top_k)

    def is_indexed(self) -> bool:
        if self.store.exists():
            if self.store.count() == 0:
                try:
                    self.store.load()
                except Exception:
                    return False
            return self.store.count() > 0
        return False
