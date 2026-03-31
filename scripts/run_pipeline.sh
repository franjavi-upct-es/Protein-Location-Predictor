#!/usr/bin/env bash
# scripts/run_pipeline.sh
# ===========================================================================
# End-to-end pipeline for the Protein Localization Predictor v2.0
# Requires: uv (https://docs.astral.sh/uv/)
#
# Usage:
#   bash scripts/run_pipeline.sh                  # Full pipeline
#   bash scripts/run_pipeline.sh --skip-download   # Skip data download
# ===========================================================================

set -euo pipefail

SKIP_DOWNLOAD=false
for arg in "$@"; do
  case $arg in
  --skip-download) SKIP_DOWNLOAD=true ;;
  *)
    echo "Unknown argument: $arg"
    exit 1
    ;;
  esac
done

echo "=========================================="
echo " Protein Localization Predictor v2.0"
echo "=========================================="
echo ""

# Step 0: Hardware detection
echo "[0/5] Detecting hardware..."
uv run python -c "
from src.utils.hardware import detect_hardware, estimate_vram_usage
from src.utils.config import load_config
cfg = load_config()
hw = detect_hardware(cfg)
print(hw.summary())
print()
est = estimate_vram_usage(
    precision=hw.precision,
    batch_size=hw.batch_size,
    gradient_checkpointing=hw.gradient_checkpointing,
)
print('Estimated VRAM usage:')
for k, v in est.items():
    print(f'  {k}: {v:.2f} GB')
"
echo ""

# Step 1: Download data
if [ "$SKIP_DOWNLOAD" = false ]; then
  echo "[1/5] Downloading data from UniProt..."
  uv run python -m src.data.download
else
  echo "[1/5] Skipping download (--skip-download)"
fi
echo ""

# Step 2: Process data
echo "[2/5] Processing and validating data..."
uv run python -m src.data.processing
echo ""

# Step 3: Split data
echo "[3/5] Creating homology-aware train/val/test splits..."
uv run python -m src.data.splitting
echo ""

# Step 4: Train model
echo "[4/5] Training model..."
uv run python -m src.training.train
echo ""

# Step 5: Evaluate
echo "[5/5] Evaluating model..."
uv run python -c "
from pathlib import Path
from src.utils.config import load_config, resolve_path

cfg = load_config()
models_dir = resolve_path(cfg, 'paths.models_dir') / 'checkpoints'

# Find best checkpoint
ckpt = None
if models_dir.exists():
    last = models_dir / 'last.ckpt'
    if last.exists():
        ckpt = last
    else:
        ckpts = sorted(models_dir.glob('*.ckpt'), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            ckpt = ckpts[0]

if ckpt is None:
    print('No checkpoint found — skipping evaluation.')
    print('Train a model first with: uv run python -m src.training.train')
    exit(0)

print(f'Evaluating checkpoint: {ckpt}')

import torch
import pandas as pd
from src.serving.predictor import Predictor
from src.data.datasets import ProteinDataset, dynamic_padding_collate
from src.evaluation.metrics import collect_predictions, compute_metrics, generate_report
from torch.utils.data import DataLoader

predictor = Predictor.from_checkpoint(ckpt, cfg)
label_list = predictor.label_list

# Load test split
splits_dir = resolve_path(cfg, 'paths.splits_dir')
test_df = pd.read_csv(splits_dir / 'test.csv')
test_df['locations'] = test_df['locations_str'].apply(lambda s: s.split('|') if isinstance(s, str) else [])

test_ds = ProteinDataset(test_df, predictor.tokenizer, label_list, max_length=predictor.max_length)
test_dl = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=dynamic_padding_collate)

results = collect_predictions(predictor.model, test_dl, device=predictor.device)
metrics = compute_metrics(results['predictions'], results['targets'], label_list)

reports_dir = resolve_path(cfg, 'paths.reports_dir')
generate_report(metrics, results['predictions'], results['targets'], label_list, reports_dir)
"
echo ""

echo "=========================================="
echo " Pipeline complete"
echo "=========================================="
