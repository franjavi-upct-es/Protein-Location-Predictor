# Makefile
# ===========================================================================
# Protein Subcellular Localization Predictor v2.0
# Managed with UV (https://docs.astral.sh/uv/)
# ===========================================================================

.PHONY: help sync sync-dev test test-fast test-cov lint format typecheck \
        quality clean download process train serve docker-build docker-run \
        hw-detect vram-estimate lock

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment management
# ---------------------------------------------------------------------------

sync: ## Install production dependencies
	uv sync --no-dev

sync-dev: ## Install all dependencies (production + development)
	uv sync
	uv run pre-commit install

lock: ## Regenerate the lockfile
	uv lock

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test: ## Run the full test suite
	uv run pytest tests/ -v --tb=short

test-fast: ## Run tests excluding slow and GPU-dependent tests
	uv run pytest tests/ -v --tb=short -m "not slow and not gpu"

test-cov: ## Run tests with coverage report
	uv run pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html:reports/coverage

lint: ## Run linter (ruff)
	uv run ruff check src/ tests/

format: ## Auto-format code (ruff)
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

typecheck: ## Run type checker (mypy)
	uv run mypy src/

quality: lint typecheck test-fast ## Run all quality checks (lint + typecheck + fast tests)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

download: ## Download protein data from UniProt
	uv run python -m src.data.download

process: ## Process and clean downloaded data
	uv run python -m src.data.processing

train: ## Train the model
	uv run python -m src.training.train

serve: ## Start the prediction API server
	uv run python -m src.serving.app

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

hw-detect: ## Detect hardware and print recommended configuration
	uv run python -c "\
	from src.utils.hardware import detect_hardware; \
	from src.utils.config import load_config; \
	cfg = load_config(); \
	hw = detect_hardware(cfg); \
	print(hw.summary())"

vram-estimate: ## Estimate VRAM usage for default configuration
	uv run python -c "\
	from src.utils.hardware import estimate_vram_usage; \
	r = estimate_vram_usage(); \
	print('\n'.join(f'{k}: {v:.2f} GB' for k,v in r.items()))"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: ## Build the Docker image
	docker build -t protein-loc-predictor:latest .

docker-run: ## Run the API server in Docker
	docker run -p 8000:8000 --gpus all protein-loc-predictor:latest

docker-run-cpu: ## Run the API server in Docker (CPU only)
	docker run -p 8000:8000 protein-loc-predictor:latest

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Remove build artifacts, caches, and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ reports/coverage/

clean-data: ## Remove all downloaded and processed data (use with caution)
	rm -rf data/raw/* data/processed/* data/splits/*

clean-all: clean clean-data ## Remove everything (artifacts + data)
	rm -rf models/* mlruns/
