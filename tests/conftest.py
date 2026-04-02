"""
Shared pytest fixtures for TuracoFlow tests.

Clears the fraud detector DB before every test so image/content hashes
from a previous run (or previous test in the same session) never bleed
through and cause false DUPLICATE_CLAIM results.
"""

import pytest
from app.api.routes.claims import _fraud


@pytest.fixture(autouse=True)
def reset_fraud_db():
    """Wipe all processed-claims records before each test."""
    _fraud.clear()
    yield
