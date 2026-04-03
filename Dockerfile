# ── Stage 1: dependencies ──────────────────────────────────────────────────────
FROM python:3.12-slim AS base

# System packages needed by EasyOCR / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying source so this layer is cached
COPY requirements.txt .

# Install CPU-only torch first — the container has no GPU, and the default
# Linux wheel pulls gigabytes of CUDA libraries unnecessarily.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies (torch is already satisfied, skipped)
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: application ───────────────────────────────────────────────────────
COPY . .

# Make the entrypoint executable
RUN chmod +x scripts/entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# The entrypoint builds the policy index on first boot if needed,
# then starts the API server.
ENTRYPOINT ["scripts/entrypoint.sh"]
