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

## [1.0.0] - Previous version

- Yeast-only data from UniProt
- Frozen ESM-2 embeddings + XGBoost classifier
- Single-label classification with sample weighting
- 64% accuracy on random train/test split
