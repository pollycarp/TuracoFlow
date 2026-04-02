"""
Lightweight numpy-based vector store.

Replaces LanceDB to avoid PyArrow/compiled DLL dependency.
For 55 policy chunks, numpy cosine similarity is instant.

Persists to disk as:
  <store_path>/vectors.npy   — float32 embeddings matrix (N x D)
  <store_path>/records.json  — metadata + content for each row
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class NumpyVectorStore:
    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self._vectors: Optional[np.ndarray] = None  # shape (N, D)
        self._records: list[dict] = []

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        self.store_path.mkdir(parents=True, exist_ok=True)
        np.save(self.store_path / "vectors.npy", self._vectors)
        with open(self.store_path / "records.json", "w", encoding="utf-8") as f:
            json.dump(self._records, f)
        logger.info(f"Saved {len(self._records)} records to {self.store_path}")

    def load(self) -> None:
        self._vectors = np.load(self.store_path / "vectors.npy")
        with open(self.store_path / "records.json", encoding="utf-8") as f:
            self._records = json.load(f)
        logger.info(f"Loaded {len(self._records)} records from {self.store_path}")

    def exists(self) -> bool:
        return (
            (self.store_path / "vectors.npy").exists()
            and (self.store_path / "records.json").exists()
        )

    def clear(self) -> None:
        self._vectors = None
        self._records = []
        # Remove persisted files if present
        for fname in ("vectors.npy", "records.json"):
            p = self.store_path / fname
            if p.exists():
                p.unlink()

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(self, records: list[dict]) -> None:
        """
        Add records to the store.
        Each record must have a 'vector' key (list[float]).
        The vector is popped and stored separately in the matrix.
        """
        if not records:
            return

        new_vectors = np.array(
            [r.pop("vector") for r in records], dtype=np.float32
        )
        meta = records  # vector already popped

        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

        self._records.extend(meta)

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        filters: Optional[dict] = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Cosine similarity search with optional exact-match metadata filters.

        Args:
            query_vector: embedding of the query
            filters: e.g. {"country": "Kenya", "product_type": "HospiCash"}
            top_k: number of results to return

        Returns:
            List of record dicts with an added 'score' key (0–1).
        """
        if self._vectors is None or len(self._records) == 0:
            return []

        # Apply metadata filters
        if filters:
            indices = [
                i for i, r in enumerate(self._records)
                if all(r.get(k) == v for k, v in filters.items())
            ]
        else:
            indices = list(range(len(self._records)))

        if not indices:
            return []

        subset = self._vectors[indices]        # (M, D)
        q = np.array(query_vector, dtype=np.float32)

        # Cosine similarity: dot(A, q) / (|A| * |q|)
        dot = subset.dot(q)                    # (M,)
        norms = np.linalg.norm(subset, axis=1) * np.linalg.norm(q)
        norms = np.maximum(norms, 1e-10)       # avoid division by zero
        similarities = dot / norms             # (M,)

        top_local = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                **self._records[indices[i]],
                "score": round(max(0.0, float(similarities[i])), 4),
            }
            for i in top_local
        ]

    def count(self) -> int:
        return len(self._records)
