# Makefile
# ===========================================================================
# Protein Subcellular Localization Predictor v2.0
# Managed with UV (https://docs.astral.sh/uv/)
# ===========================================================================

.PHONY: help sync sync-dev test test-fast test-cov lint format typecheck \
        quality clean download process train serve docker-build docker-run \
        hw-detect vram-estimate qlora-smoke sdpa-smoke lock auto-batch-size \
				smoke

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

qlora-smoke: ## Validate the QLoRA path on the configured backbone (requires CUDA + bitsandbytes)
	uv run python -c "\
	import sys; \
	import torch; \
	from src.models.esm_lora import build_esm_lora_backbone, get_quantization_runtime_issue; \
	from src.utils.config import load_config; \
	issue = get_quantization_runtime_issue(); \
	issue and (print(f'QLoRA smoke skipped: {issue}'), sys.exit(0)); \
	cfg = load_config(overrides=['model.quantization.enabled=true']); \
	m = build_esm_lora_backbone(cfg, enable_gradient_checkpointing=True).to('cuda'); \
	print('Quantized backbone loaded OK on', next(m.parameters()).device); \
	allocated_gb = torch.cuda.memory_allocated() / (1024**3); \
	print(f'VRAM allocated after load: {allocated_gb:.2f} GB')"

sdpa-smoke: ## Validate the SDPA patch loads and produces equivalent outputs
	uv run python -c "\
	import torch; \
	from transformers import EsmModel, AutoTokenizer; \
	from src.models.sdpa_patch import patch_esm_sdpa, unpatch_esm_sdpa; \
	name = 'facebook/esm2_t6_8M_UR50D'; \
	tok = AutoTokenizer.from_pretrained(name); \
	enc = tok(['MSKGEELFTGVVPILVELDG', 'MQIFVKTLTGKTITLEVEPSDT'], return_tensors='pt', padding=True); \
	unpatch_esm_sdpa(); \
	ref_model = EsmModel.from_pretrained(name, add_pooling_layer=False).eval(); \
	ref = ref_model(**enc).last_hidden_state; \
	patch_esm_sdpa(); \
	sdpa_model = EsmModel.from_pretrained(name, add_pooling_layer=False).eval(); \
	out = sdpa_model(**enc).last_hidden_state; \
	diff = (ref - out).abs().max().item(); \
	print(f'Max abs diff stock vs SDPA: {diff:.2e}'); \
	assert diff < 1e-4, 'SDPA patch is not numerically equivalent'; \
	print('SDPA patch verified OK')"

auto-batch-size: ## Probe the largest batch size that fits on the current GPU
	uv run python -m src.training.auto_batch_size

smoke: ## Run the slow end-to-end smoke test on synthetic data
	uv run pytest tests/integration/test_end_to_end_smoke.py -v -m slow

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
