#!/bin/bash
set -e

# Build the policy index on first boot if the vector store is missing.
# On subsequent starts the index is already on the mounted volume and
# this step is skipped instantly.
if [ ! -f /data/vector_store/vectors.npy ]; then
    echo "Policy index not found — building now (first boot only)..."
    LANCEDB_PATH=/data/vector_store python scripts/build_index.py
    echo "Policy index built."
else
    echo "Policy index found — skipping build."
fi

# Start the API server
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
