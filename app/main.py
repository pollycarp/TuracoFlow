import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.routes import claims
from app.models.schemas import HealthResponse
from app.modules.rag import PolicyRAG

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_rag = PolicyRAG()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the policy index on startup."""
    if _rag.is_indexed():
        logger.info("Policy index found — loading into memory.")
        _rag.store.load()
    else:
        logger.warning("Policy index not found. Run: python scripts/build_index.py")
    yield


app = FastAPI(
    title="TuracoFlow",
    description="Automated WhatsApp Claims Validator — Turaco Insurance",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(claims.router)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        service="TuracoFlow",
        index_ready=_rag.is_indexed(),
    )
