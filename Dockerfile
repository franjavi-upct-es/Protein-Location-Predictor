# Dockerfile
# ===========================================================================
# Multi-stage build for the Protein Localization Predictor v2.0
# Uses UV for fast, reproducible dependency management.
#
# Usage:
#   docker build -t protein-loc-predictor:latest .
#   docker run -p 8000:8000 --gpus all protein-loc-predictor:latest
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base with UV and system dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Install UV (official one-liner from docs.astral.sh)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System dependencies for BioPython, MMseqs2, and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  git \
  && rm -rf /var/lib/apt/lists/*

# Install MMseqs2 for homology-aware splitting
RUN curl -fsSL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz | \
  tar xz -C /usr/local/bin --strip-components=2 mmseqs/bin/mmseqs || \
  echo "WARN: MMseqs2 installation failed — will use random splitting fallback"

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2: Dependencies (cached layer)
# ---------------------------------------------------------------------------
FROM base AS deps

# Copy only dependency files first (for Docker layer caching)
COPY pyproject.toml uv.lock* .python-version ./

# Sync production dependencies into the project venv
# --frozen ensures the lockfile is used exactly as-is
# --no-dev excludes development dependencies
RUN uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------------------
# Stage 3: Production
# ---------------------------------------------------------------------------
FROM deps AS production

# Copy the rest of the project
COPY src/ src/
COPY configs/ configs/
COPY README.md ./

# Install the project itself (now that source is available)
RUN uv sync --frozen --no-dev

# Create directories for runtime data (mounted as volumes in practice)
RUN mkdir -p data/raw data/processed data/splits models mlruns reports/figures

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Use uv run to execute within the managed environment
CMD ["uv", "run", "uvicorn", "src.serving.app:app", \
  "--host", "0.0.0.0", "--port", "8000"]
