# Sprint 6 — Comparison report

Headline metrics across baselines, the trained model, and the external DeepLoc 2.0 benchmark.

## Test-set metrics

| Model | f1_macro | f1_micro | precision_macro | recall_macro | exact_match_ratio | hamming_loss |
|---|---:|---:|---:|---:|---:|---:|
| Linear probe (frozen ESM-2) | 0.511 | 0.656 | 0.423 | 0.828 | 0.330 | 0.137 |
| XGBoost (frozen ESM-2, v1.0 replica) | 0.765 | 0.833 | 0.869 | 0.702 | 0.661 | 0.051 |
| Trained model (this project) | not available | not available | not available | not available | not available | not available |

## External benchmark — DeepLoc 2.0

Evaluated on **2 of 3 sequences** from the packaged DeepLoc 2.0 demo set, using DeepLoc's own predictions as the reference labels.
1 sequences were dropped because their DeepLoc labels do not exist in this project's taxonomy.

| Model | f1_macro | f1_micro | precision_macro | recall_macro | exact_match_ratio | hamming_loss |
|---|---:|---:|---:|---:|---:|---:|
| Trained model on DeepLoc test set | 0.143 | 0.333 | 0.143 | 0.143 | 0.000 | 0.286 |

## Linear probe — per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cytoplasm | 0.712 | 0.836 | 0.769 | 2045 |
| Endoplasmic Reticulum | 0.281 | 0.769 | 0.412 | 446 |
| Golgi Apparatus | 0.190 | 0.808 | 0.308 | 307 |
| Membrane | 0.755 | 0.824 | 0.788 | 1836 |
| Mitochondrion | 0.434 | 0.784 | 0.559 | 528 |
| Nucleus | 0.729 | 0.858 | 0.788 | 1893 |
| Peroxisome | 0.082 | 0.828 | 0.149 | 29 |
| Secreted/Extracellular | 0.552 | 0.917 | 0.690 | 545 |
| Vacuole | 0.075 | 0.830 | 0.137 | 53 |

## XGBoost — per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cytoplasm | 0.797 | 0.849 | 0.822 | 2045 |
| Endoplasmic Reticulum | 0.814 | 0.659 | 0.729 | 446 |
| Golgi Apparatus | 0.828 | 0.612 | 0.704 | 307 |
| Membrane | 0.901 | 0.846 | 0.873 | 1836 |
| Mitochondrion | 0.902 | 0.712 | 0.796 | 528 |
| Nucleus | 0.862 | 0.855 | 0.858 | 1893 |
| Peroxisome | 1.000 | 0.483 | 0.651 | 29 |
| Secreted/Extracellular | 0.864 | 0.850 | 0.857 | 545 |
| Vacuole | 0.857 | 0.453 | 0.593 | 53 |

---

### Files consumed

- Linear probe: `/home/pyros05/Escritorio/Protein-Location-Predictor/reports/baselines/linear_probe.json` (found)
- XGBoost: `/home/pyros05/Escritorio/Protein-Location-Predictor/reports/baselines/xgboost_baseline.json` (found)
- Trained model: `/home/pyros05/Escritorio/Protein-Location-Predictor/reports/evaluation_report.json` (missing)
- DeepLoc benchmark: `/home/pyros05/Escritorio/Protein-Location-Predictor/reports/benchmarks/deeploc.json` (found)
