All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Unreleased

### Sprint 1 — Foundation and Configuration

#### Added

- Project skeleton with proper Python packaging (`pyproject.toml`)
- Layered YAML configuration system (`configs/base.yaml`, `training.yaml`, `inference.yaml`)
- Hardware auto-detection with GPU profiling and VRAM estimation (`src/utils/hardware.py`)
- Structured logging with console and file output (`src/utils/logging.py`)
- Reproducibility utilities with full seed management (`src/utils/reproducibility.py`)
- GPU profile database for NVIDIA Blackwell, Ada Lovelace, Ampere, Turing, and Apple Silicon
- VRAM-based fallback tiers for unknown hardware
- Environment variable overrides (`PROT_LOC_` prefix) and CLI override support
- Comprehensive test suite for config, hardware, logging, and reproducibility
- Makefile with targets for install, test, lint, format, typecheck, and Docker
- Dockerfile with multi-stage build
- Docker Compose configuration with optional MLflow tracking
- Pre-commit hooks (ruff, trailing whitespace, YAML/TOML validation)
- Placeholder modules for all Sprint 2-4 components with documented interfaces

### Sprint 2 — Data Pipeline

#### Added

- Multi-species UniProt download with cursor-based pagination (`src/data/download.py`)
  - Configurable organism sources (yeast, human, mouse out of the box)
  - Automatic column normalization and deduplication
- Multi-label location annotation pipeline (`src/data/processing.py`)
  - Full UniProt location string parser with evidence code removal
  - Topology annotation filtering (single-pass, multi-pass, peripheral)
  - Configurable hierarchical location mapping from YAML
  - Sequence validation (alphabet, length, non-standard amino acids)
  - Class frequency filtering with multi-label awareness
- Homology-aware train/val/test splitting (`src/data/splitting.py`)
  - MMseqs2 clustering at configurable sequence identity threshold
  - Cluster-level split assignment (no cluster spans multiple splits)
  - Automatic fallback to stratified random splitting when MMseqs2 unavailable
- Data validation framework (`src/data/validation.py`)
  - Raw data schema and integrity checks
  - Processed data distribution analysis
  - Split leakage detection (accession overlap between splits)
  - ValidationReport dataclass with structured error/warning/stats output
- PyTorch Dataset and Lightning DataModule (`src/data/datasets.py`)
  - On-the-fly ESM-2 tokenization with dynamic padding collation
  - Multi-hot label encoding for multi-label classification
  - Optional external feature vector integration
  - Memory-efficient batch collation (pad to longest in batch, not max_length)
- 60 new unit tests covering processing, splitting, validation, and datasets
- Migrated from pip/setuptools to UV package manager
  - `pyproject.toml` with `[dependency-groups]` and hatchling build backend
  - `.python-version` file for UV Python version management
  - Updated Makefile: all commands use `uv run` / `uv sync`
  - Updated Dockerfile: UV-based multi-stage build with layer caching
  - Updated pipeline script: `uv run` throughout

#### Changed

- Build backend from setuptools to hatchling (UV recommended)
- Minimum Python version from 3.10 to 3.11
- Makefile targets: `install` → `sync`, `install-dev` → `sync-dev`

### Sprint 3 — Model and Training

#### Added

- ESM-2 + LoRA backbone module (`src/models/esm_lora.py`)
  - PEFT-based LoRA injection into query/key/value/dense attention layers
  - Gradient checkpointing support for memory efficiency
  - Three pooling strategies: `mean`, `cls`, `mean_cls` (concatenated)
  - Attention mask-aware mean pooling (padding tokens excluded)
- Multi-label classifier head (`src/models/classifier_head.py`)
  - MLP with LayerNorm, configurable hidden dims, GELU/ReLU activation
  - Raw logit output (no sigmoid — handled by loss for numerical stability)
  - Xavier initialization on the final layer for stable early training
  - Factory method `from_config()` for YAML-driven construction
- Loss functions (`src/models/losses.py`)
  - FocalLoss: per-class weighted BCE with (1-p_t)^gamma modulation
  - Auto-computed alpha weights from class frequency (inverse frequency)
  - HierarchicalLoss: biological distance penalty for false positives
  - Built-in distance matrix encoding compartment relationships
  - CombinedLoss: weighted sum with per-component logging dict
- PyTorch Lightning training module (`src/models/lightning_module.py`)
  - Wraps ESM-2 + LoRA + classifier in a single trainable module
  - Differential learning rates (lower for backbone LoRA, higher for head)
  - OneCycleLR scheduler with cosine annealing and warmup
  - Per-class and macro F1/Precision/Recall via torchmetrics
  - Per-class F1 logged at validation epoch end for class-level monitoring
- Custom callbacks (`src/training/callbacks.py`)
  - VRAMMonitorCallback: logs allocated/reserved/peak GPU memory
  - GradientNormCallback: separate backbone vs head gradient norm tracking
- Training entry point (`src/training/train.py`)
  - Full orchestration: config → hardware detect → data → model → train → test
  - Auto-discovers label set and class frequencies from training split
  - Applies hardware-recommended batch size, precision, grad checkpointing
  - MLflow experiment tracking with CSV logger fallback
  - EarlyStopping on val/f1_macro, top-k checkpoint saving
  - CLI with `--overrides` for runtime config changes
- 32 new unit tests covering classifier head, losses, pooling, and ESM utilities

### Sprint 4 — Features, Evaluation, and Serving

#### Added

- Biophysical feature computation (`src/features/biophysical.py`)
  - Molecular weight, isoelectric point, GRAVY, aromaticity, instability index
  - Secondary structure fractions (helix, turn, sheet) via BioPython
  - Automatic standardization (zero mean, unit variance) for model compatibility
  - NaN handling with column-mean imputation for occasional failures
- Signal peptide prediction wrapper (`src/features/signal_peptide.py`)
  - SignalP 6.0 integration with FASTA batch processing
  - Returns binary has_signal_peptide and probability score
  - Graceful degradation to zero vectors when SignalP is not installed
  - Config-driven enable/disable toggle
- Transmembrane topology wrapper (`src/features/transmembrane.py`)
  - TMHMM 2.0 / DeepTMHMM integration
  - Returns TM helix count, binary flag, and fraction of residues in membrane
  - Graceful degradation to zero vectors when TMHMM is not installed
- Evaluation module (`src/evaluation/metrics.py`)
  - Multi-label metrics: macro/micro/weighted F1, precision, recall, hamming loss
  - Exact match ratio for multi-label evaluation
  - Per-class precision, recall, F1, and support counts
  - Formatted classification report (text)
  - Per-class F1 bar chart with support annotations
  - Multi-label confusion co-occurrence heatmap
  - Full report generation (text + CSV + figures)
- FastAPI REST service (`src/serving/app.py`)
  - POST /predict: single sequence prediction
  - POST /predict/batch: batch prediction (up to 100 sequences)
  - GET /health: health check with model status and device info
  - GET /labels: list of supported location classes
  - Async lifespan with model loading on startup and warmup
  - Graceful degraded mode when no checkpoint is available
  - CORS middleware enabled
- Pydantic request/response schemas (`src/serving/schemas.py`)
  - Input validation: amino acid alphabet check, length bounds
  - Automatic uppercasing and whitespace stripping
  - Confidence score bounds enforcement [0, 1]
- Inference predictor (`src/serving/predictor.py`)
  - Checkpoint loading with automatic device detection
  - Single and batch prediction with configurable threshold and top-k
  - Always returns at least one prediction (highest confidence if nothing passes threshold)
  - Model warmup method for eliminating cold-start latency
- 50 new tests: biophysical features, external tool degradation, evaluation metrics,
  API schemas, API integration tests (health, predict, batch, labels endpoints)

### Sprint 5 — Production Hardening (in progress)

#### Fixed

- `src/data/splitting.py`: `_run_mmseqs2_clustering` now correctly iterates
  over `cluster_map.items()` instead of unpacking dict keys, which would
  raise `ValueError` on the first successful MMseqs2 run.
- `src/evaluation/metrics.py`: `collect_predictions` no longer passes a
  broken `.get("to", lambda _: None) or None` expression for external
  features. Replaced with explicit `None`-aware device transfer so that
  the biophysical features pipeline works at evaluation time.
- `src/models/esm_lora.py`: log message now reads `LoRA rank=8, alpha=16`
  instead of the malformed `LoRA rank8, alpha=16`.
- `pyproject.toml`: uncommented the `[tool.uv.sources]` table header so
  the `torch = { index = "pytorch-cu124" }` source declaration is parsed
  into the correct table.

#### Added

- Optional QLoRA path in `src/models/esm_lora.py`. When
  `model.quantization.enabled` is true, the ESM-2 backbone is loaded in
  4-bit (NF4 by default) via bitsandbytes and prepared for k-bit
  training before LoRA is injected. CPU and non-quantized paths are
  unchanged.
- New configuration section `model.quantization` in `configs/base.yaml`
  with sensible defaults (NF4, bfloat16 compute dtype, double quant).
  Disabled by default to keep existing tests and CPU-only environments
  working out of the box.
- `bitsandbytes>=0.45.0` added to project dependencies.
- New tests `tests/unit/test_esm_lora_quantization.py`:
  - Pure validation tests for `_build_bnb_config` (no GPU required).
  - GPU-marked smoke test that loads `facebook/esm2_t6_8M_UR50D` in
    4-bit, applies LoRA, and runs a forward pass end-to-end.
- New `make qlora-smoke` target that loads the configured backbone
  with quantization enabled and reports VRAM usage after load.
- Optional SDPA / Flash Attention 2 path for ESM-2 via a monkey-patch
  in `src/models/sdpa_patch.py`. The patch replaces
  `EsmSelfAttention.forward` with a version that calls
  `torch.nn.functional.scaled_dot_product_attention`, which
  transparently dispatches to Flash Attention 2 on Ampere/Ada/Blackwell
  GPUs and reduces activation memory ~30-40% on long sequences. The
  patch is idempotent, opt-in via `model.backbone.use_sdpa_attention`,
  and falls back to the stock forward in any case it cannot represent
  exactly (`output_attentions=True`, cross-attention, past KV cache,
  head masks).
- New tests `tests/unit/test_sdpa_patch.py`:
  - Patch lifecycle tests (apply, idempotent, unpatch).
  - Numerical equivalence test that runs the smallest ESM-2 variant
    through both paths and asserts max absolute difference < 1e-4.
  - Fallback test for `output_attentions=True`.
- New `make sdpa-smoke` target that verifies the patch loads and
  produces numerically equivalent outputs on the host installation.

- New `LengthBucketBatchSampler` in `src/data/samplers.py`. Groups
  protein sequences of similar length into the same training batch to
  minimize padding waste, which is a significant compute drain on
  heavy-tailed length distributions like UniProt. Opt-in via
  `training.use_length_bucketing`. Wired into `ProteinDataModule.train_dataloader`
  with a clean fallback to the previous random-shuffled path. Validation
  and test loaders are unchanged. Tests in `tests/unit/test_samplers.py`
  verify coverage, determinism, and that bucketing reduces padding waste
  by at least 4× on a power-law length distribution.

- Empirical batch-size auto-tuning in `src/training/auto_batch_size.py`.
  When `training.auto_batch_size: true` and a CUDA GPU is available,
  the trainer probes batch sizes from largest to smallest by running a
  forward+backward on a synthetic batch and catching
  `torch.cuda.OutOfMemoryError`. The result is cached in
  `.cache/auto_batch_size.json` keyed by a hash of the relevant
  configuration fields, so subsequent runs are instant. Also exposed
  as a standalone CLI: `make auto-batch-size`.

- New `RuntimeFingerprintCallback` in `src/training/runtime_fingerprint.py`.
  Captures GPU + driver + CUDA version, package versions, git commit
  SHA + dirty flag, SHA-256 of the resolved configuration, and SHA-256
  of each split CSV at training start. Persisted as JSON under
  `reports/fingerprints/<run_id>.json` and the most relevant fields are
  also logged to MLflow as hyperparameters. Failures during fingerprint
  capture are logged but never block training. Registered automatically
  in `_build_callbacks` so every training run is reproducible.

- New end-to-end smoke test in `tests/integration/test_end_to_end_smoke.py`.
  Marked `@pytest.mark.slow`. Generates 60 synthetic protein sequences
  with synthetic location labels, builds a minimal in-memory config
  pointing at `facebook/esm2_t6_8M_UR50D`, runs the full pipeline
  (DataModule → LightningModule → Trainer.fit for 2 epochs → forward
  pass on the test loader). Asserts that the trainer steps at least
  once and that the trained model produces logits of the correct shape.
  Available as `make smoke`.

### Sprint 6 — Trustworthy Results (in progress)

#### Added

- **Pydantic v2 configuration schema** in `src/utils/config_schema.py`.
  Defines an explicit type and range constraint for every section the
  project authors. The new `validate_config()` entry point catches
  typos like `lora.rnak: 8` and out-of-range values like negative
  learning rates before training starts. Wired into `train.py:main()`
  via `load_config(..., validate=True)`. Existing tests use
  `validate=False` (the default) to keep working with intentionally
  minimal config dicts. New tests in `tests/unit/test_config_schema.py`.
- `pydantic>=2.5.0` added to project dependencies.

- **Frozen ESM-2 + linear probe baseline** in
  `src/baselines/linear_probe.py`. Loads ESM-2 without LoRA, mean-pools
  per-protein embeddings, fits a one-vs-rest logistic regression with
  balanced class weights, and writes
  `reports/baselines/linear_probe.json`. This is the lower-bound
  baseline: any fine-tuned model that does not beat it has a problem.
  Available as `make linear-probe`.

- **Embedding cache** in `src/baselines/embedding_cache.py` shared by
  every frozen-backbone baseline. Caches per-split embeddings on disk
  under `.cache/embeddings/`, keyed by `(backbone, pooling, max_length,
split file mtime+size)`. A re-download of the splits invalidates the
  cache automatically. The XGBoost and linear probe baselines reuse the
  same embeddings transparently.

- **Frozen ESM-2 + XGBoost baseline** in
  `src/baselines/xgboost_baseline.py`. Replicates the v1.0 architecture
  on the v2.0 multi-species, homology-aware splits to give an
  apples-to-apples comparison with v1.0 published numbers. Trains one
  classifier per class with `scale_pos_weight` matched to class
  frequencies. Writes `reports/baselines/xgboost_baseline.json`.
  Available as `make xgboost-baseline`. XGBoost is now in the optional
  `baselines` dependency group: `uv sync --group baselines`.

- **DeepLoc 2.0 benchmark integration** in
  `src/baselines/deeploc_benchmark.py`. Loads the DeepLoc 2.0 test set
  from `benchmarks/deeploc/`, maps DeepLoc location names to project
  classes via a configurable label map, runs the latest checkpoint on
  every sequence, and writes `reports/benchmarks/deeploc.json`. The
  dataset is not bundled (it has its own license terms); see
  `benchmarks/deeploc/README.md` for instructions on how to obtain it.
  Available as `make deeploc-benchmark`.

- **Per-organism evaluation breakdown** in
  `src/evaluation/per_organism.py`. Slices predictions by `organism_id`
  and computes `compute_metrics` independently per slice, with pretty
  organism names for the common UniProt taxonomy IDs. Helps detect
  silent regressions on yeast (the v1.0 dominant organism). Skips
  organisms with fewer than `min_samples` rows.

- **Per-class threshold tuning** in
  `src/evaluation/threshold_tuning.py`. Sweeps thresholds in
  [0.05, 0.95] per class on the validation set to maximize F1, then
  persists the result as `models/checkpoints/thresholds.json`. The
  `Predictor` loads this file at startup and uses per-class thresholds
  in `predict()` instead of the global `threshold=0.5`. Wired into
  `train.py` so every training run automatically produces tuned
  thresholds. Tests in `tests/unit/test_sprint6_components.py`.

- **Comparison report generator** in
  `src/evaluation/comparison_report.py`. Reads every JSON metrics file
  produced during the sprint (linear probe, XGBoost, DeepLoc, trained
  model) and emits a single Markdown report at
  `reports/sprint-6-comparison.md` comparing them on the headline
  metrics, plus per-class breakdowns when available. Missing inputs
  are flagged as "not available" so the report makes it obvious what
  is still pending. Available as `make comparison-report`.

- New Makefile targets: `linear-probe`, `xgboost-baseline`,
  `deeploc-benchmark`, `comparison-report`, `baselines` (runs both
  baselines back-to-back).

## [1.0.0] - Previous version

- Yeast-only data from UniProt
- Frozen ESM-2 embeddings + XGBoost classifier
- Single-label classification with sample weighting
- 64% accuracy on random train/test split
