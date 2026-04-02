"""
Unit tests for the RAG pipeline (Phase 3).

Tests verify:
1. Index builds from policy files without errors
2. Correct policy retrieved for a Kenya malaria query
3. Country filter prevents Nigeria results appearing for Kenya query
4. Claim limit information is retrievable
5. Re-indexing is idempotent (no duplicate data)
"""

import pytest
from pathlib import Path
from app.modules.rag import PolicyRAG, _parse_policy_metadata, _chunk_policy


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rag():
    """Build the index once for all tests in this module."""
    r = PolicyRAG()
    r.build_index()
    return r


# ── Metadata parser tests ──────────────────────────────────────────────────────

def test_parse_metadata_kenya():
    policies_dir = Path(__file__).parent.parent / "data" / "policies"
    text = (policies_dir / "kenya_hospicash.txt").read_text(encoding="utf-8")
    meta = _parse_policy_metadata(text, "kenya_hospicash")
    assert meta["country"] == "Kenya"
    assert meta["product_type"] == "HospiCash"
    assert "TUR-KE-HC" in meta["policy_code"]


def test_parse_metadata_uganda():
    policies_dir = Path(__file__).parent.parent / "data" / "policies"
    text = (policies_dir / "uganda_personal_accident.txt").read_text(encoding="utf-8")
    meta = _parse_policy_metadata(text, "uganda_personal_accident")
    assert meta["country"] == "Uganda"
    assert meta["product_type"] == "Personal Accident"


def test_parse_metadata_nigeria():
    policies_dir = Path(__file__).parent.parent / "data" / "policies"
    text = (policies_dir / "nigeria_hospicash.txt").read_text(encoding="utf-8")
    meta = _parse_policy_metadata(text, "nigeria_hospicash")
    assert meta["country"] == "Nigeria"
    assert meta["product_type"] == "HospiCash"


# ── Chunker tests ──────────────────────────────────────────────────────────────

def test_chunker_produces_multiple_sections():
    policies_dir = Path(__file__).parent.parent / "data" / "policies"
    text = (policies_dir / "kenya_hospicash.txt").read_text(encoding="utf-8")
    meta = _parse_policy_metadata(text, "kenya_hospicash")
    chunks = _chunk_policy(text, meta)
    assert len(chunks) >= 4, "Expected at least 4 sections in Kenya HospiCash"


def test_chunker_preserves_metadata_in_all_chunks():
    policies_dir = Path(__file__).parent.parent / "data" / "policies"
    text = (policies_dir / "uganda_hospicash.txt").read_text(encoding="utf-8")
    meta = _parse_policy_metadata(text, "uganda_hospicash")
    chunks = _chunk_policy(text, meta)
    for chunk in chunks:
        assert chunk["country"] == "Uganda"
        assert chunk["product_type"] == "HospiCash"
        assert "content" in chunk
        assert len(chunk["content"]) > 0


# ── Index build tests ──────────────────────────────────────────────────────────

def test_index_builds_successfully(rag):
    assert rag.is_indexed(), "Index should be marked as built after build_index()"


def test_reindex_is_idempotent(rag):
    """Re-building the index should not duplicate records."""
    count_before = rag.store.count()
    rag.build_index()
    count_after = rag.store.count()
    assert count_before == count_after, "Re-indexing must not create duplicate records"


# ── Retrieval tests ────────────────────────────────────────────────────────────

def test_retrieve_returns_results(rag):
    results = rag.retrieve("what conditions are covered for inpatient hospital stay")
    assert len(results) > 0, "Should return at least one result"


def test_retrieve_kenya_malaria_returns_kenya_policy(rag):
    results = rag.retrieve(
        query="Is malaria covered for inpatient hospital stay?",
        country="Kenya",
        top_k=3,
    )
    assert len(results) > 0
    countries = {r["country"] for r in results}
    assert countries == {"Kenya"}, f"Expected only Kenya results, got: {countries}"


def test_retrieve_nigeria_does_not_return_kenya(rag):
    results = rag.retrieve(
        query="HospiCash daily benefit amount",
        country="Nigeria",
        top_k=3,
    )
    for r in results:
        assert r["country"] == "Nigeria", (
            f"Country filter failed — got a {r['country']} result in a Nigeria query"
        )


def test_retrieve_kenya_claim_limits(rag):
    results = rag.retrieve(
        query="maximum claim amount per single event",
        country="Kenya",
        product_type="HospiCash",
        top_k=3,
    )
    assert len(results) > 0
    combined_content = " ".join(r["content"] for r in results).lower()
    assert "10,000" in combined_content or "10 nights" in combined_content, (
        "Kenya HospiCash claim limit (KES 10,000 / 10 nights) should appear in results"
    )


def test_retrieve_uganda_personal_accident(rag):
    results = rag.retrieve(
        query="disability benefit boda-boda accident",
        country="Uganda",
        product_type="Personal Accident",
        top_k=3,
    )
    assert len(results) > 0
    combined = " ".join(r["content"] for r in results).lower()
    assert "accident" in combined or "disability" in combined


def test_retrieve_score_is_between_0_and_1(rag):
    results = rag.retrieve("hospital stay coverage", top_k=5)
    for r in results:
        # Score is clipped cosine similarity — always in [0, 1]
        assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"
