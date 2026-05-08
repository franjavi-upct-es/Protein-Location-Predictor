# Subcellular Protein Localization Predictor v2.0

An end-to-end deep learning system for predicting protein subcellular localization from amino acid sequences, built with ESM-2 and LoRA fine-tuning.

## What's new in v2.0

This is a ground-up rewrite of the [v1.0 prototype](https://github.com/franjavi-upct-es/Protein-Location-Predictor/tree/cfdbcdc571f0c3ee759eee45198457a68c1edaa1). Key improvements:

| Area | v1.0 | v2.0 |
|------|------|------|
| **Data** | Yeast only, random splits | Multi-species, homology-aware splits |
| **Model** | Frozen ESM-2 → XGBoost | ESM-2 + LoRA fine-tuning (all 33 layers) |
| **Classification** | Single-label, flat | Multi-label, hierarchical |
| **Loss** | Cross-entropy | Focal Loss + Hierarchical penalty |
| **Hardware** | Manual config, CUDA-only | Auto-detection, dynamic batch sizing |
| **Code** | Scripts with hardcoded paths | Proper Python package, YAML config, tests |
| **Tracking** | Print statements | MLflow + structured logging |

## Quick start

```bash
# Clone and install
git clone https://github.com/franjavi-upct-es/Protein-Location-Predictor.git
cd Protein-Location-Predictor
uv sync

# Detect your hardware
make hw-detect

# Run tests
make test

# Run the full pipeline (when implemented)
bash scripts/run_pipeline.sh
```

## Project structure

```
Protein-Location-Predictor/
├── configs/                    # YAML configuration files
│   ├── base.yaml               #   Shared settings, paths, model config
│   ├── training.yaml            #   Training-specific overrides
│   ├── inference.yaml           #   Serving-specific overrides
│   └── hardware/
│       └── gpu_profiles.yaml    #   GPU-specific batch size and precision
├── src/
│   ├── data/                   # Data pipeline
│   │   ├── download.py          #   UniProt multi-species download
│   │   ├── processing.py        #   Cleaning, multi-label annotation
│   │   ├── splitting.py         #   Homology-aware train/val/test splits
│   │   ├── validation.py        #   Schema and integrity checks
│   │   ├── datasets.py          #   PyTorch Dataset and DataModule
│   │   └── external_features.py #   SignalP, TMHMM wrappers
│   ├── models/                 # Model architecture
│   │   ├── esm_lora.py          #   ESM-2 backbone with LoRA adapters
│   │   ├── classifier_head.py   #   Multi-label classification MLP
│   │   ├── lightning_module.py  #   PyTorch Lightning training module
│   │   └── losses.py            #   Focal Loss + Hierarchical Loss
│   ├── features/               # Feature engineering
│   │   ├── biophysical.py       #   MW, pI, GRAVY, aromaticity
│   │   ├── signal_peptide.py    #   SignalP wrapper
│   │   └── transmembrane.py     #   TMHMM wrapper
│   ├── training/               # Training orchestration
│   │   ├── train.py             #   Main training entry point
│   │   └── callbacks.py         #   Custom Lightning callbacks
│   ├── serving/                # API and inference
│   │   ├── app.py               #   FastAPI application
│   │   ├── schemas.py           #   Request/response models
│   │   └── predictor.py         #   Inference-time predictor
│   └── utils/                  # Shared utilities
│       ├── config.py            #   YAML config loading and merging
│       ├── logging.py           #   Structured logging
│       ├── hardware.py          #   GPU detection and VRAM profiling
│       └── reproducibility.py   #   Seed management
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── scripts/
│   └── run_pipeline.sh         # End-to-end pipeline runner
├── Makefile                    # Project commands
├── Dockerfile                  # Container build
├── docker-compose.yaml         # Local development stack
├── pyproject.toml              # Package definition and tool config
└── .pre-commit-config.yaml     # Code quality hooks
```

## Configuration

All settings are managed through YAML files in `configs/`. The system uses a layered merge strategy:

```
base.yaml  ←  training.yaml / inference.yaml  ←  ENV vars  ←  CLI overrides
```

Override any value via environment variables (prefix `PROT_LOC_`, double underscore for nesting):

```bash
export PROT_LOC_TRAINING__BATCH_SIZE=4
export PROT_LOC_MODEL__LORA__RANK=16
```

Or via CLI overrides:

```python
from src.utils.config import load_config
cfg = load_config(overrides=["model.lora.rank=16", "training.max_epochs=10"])
```

## Hardware-aware training

The system automatically detects your GPU and configures batch size, precision, and memory optimization strategy. Supported hardware:

- **NVIDIA Blackwell** (RTX 50xx): Native bfloat16, optimal performance
- **NVIDIA Ada Lovelace** (RTX 40xx): bfloat16 supported
- **NVIDIA Ampere** (RTX 30xx): bfloat16 supported
- **NVIDIA Turing** (RTX 20xx): fp16 fallback
- **Apple Silicon** (M1/M2): fp16 via MPS backend
- **CPU**: fp32 fallback (slow but functional)

For an 8 GB GPU (e.g., RTX 5060), the default configuration uses:
- bfloat16 mixed precision
- Gradient checkpointing (~40% activation memory reduction)
- Batch size 2
- ~6.5 GB estimated VRAM usage

Run `make hw-detect` to see your resolved profile, or `make vram-estimate` for a memory breakdown.

## Results

### Model Performance vs. Baselines

The v2.0 model (ESM-2 + LoRA) significantly outperforms the linear probe baseline and achieves competitive results compared to the v1.0 XGBoost replica while providing better recall across rare classes.

| Model | F1 (Macro) | F1 (Micro) | Precision (Macro) | Recall (Macro) | Exact Match |
|-------|------------|------------|-------------------|----------------|-------------|
| **v2.0 (LoRA Fine-tuned)** | **0.792** | 0.817 | 0.842 | 0.750 | 0.651 |
| XGBoost (v1.0 replica) | 0.765 | **0.833** | **0.869** | 0.702 | **0.661** |
| Linear Probe | 0.511 | 0.656 | 0.423 | **0.828** | 0.330 |

### Per-Class Performance (v2.0)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Cytoplasm | 0.809 | 0.794 | 0.801 | 2045 |
| Membrane | 0.876 | 0.814 | 0.844 | 1836 |
| Nucleus | 0.894 | 0.767 | 0.825 | 1893 |
| Secreted/Extracellular | 0.862 | 0.861 | 0.861 | 545 |
| Mitochondrion | 0.900 | 0.780 | 0.836 | 528 |
| Endoplasmic Reticulum | 0.781 | 0.702 | 0.739 | 446 |
| Golgi Apparatus | 0.766 | 0.704 | 0.733 | 307 |
| Peroxisome | 0.875 | 0.724 | 0.792 | 29 |
| Vacuole | 0.821 | 0.604 | 0.696 | 53 |

Detailed visualizations including the [Confusion Matrix](reports/figures/confusion_matrix.png) and [F1 Scores by Class](reports/figures/f1_scores_by_class.png) can be found in the `reports/` directory.

## Development

```bash
# Install with dev dependencies
make install-dev

# Run tests
make test          # Full suite
make test-fast     # Skip slow/GPU tests

# Code quality
make lint          # Ruff linter
make format        # Auto-format
make typecheck     # Mypy
make quality       # All of the above
```

## License

MIT
