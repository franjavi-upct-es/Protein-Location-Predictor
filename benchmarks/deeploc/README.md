# DeepLoc 2.0 benchmark

This directory holds the DeepLoc 2.0 test set used by
`src.baselines.deeploc_benchmark`. The data is **not** included in this
repository because the upstream dataset has its own license terms.

## What to put here

Two files:

- `test.fasta` — protein sequences in FASTA format
- `test_labels.tsv` — tab-separated, two columns: accession and the
  pipe-separated list of subcellular locations

Example `test_labels.tsv` row:

```
P12345 Nucleus|Cytoplasm
```

## How to obtain the dataset

1. Visit <https://services.healthtech.dtu.dk/services/DeepLoc-2.0/>
2. Request access to the dataset (academic use is typically free).
3. Download the test set distribution.
4. Place the FASTA and label files into this directory with the names
   above. If the files are named differently, just rename them.

## How to run the benchmark

```bash
uv run python -m src.baselines.deeploc_benchmark
```

The script picks the latest checkpoint under `models/checkpoints/` and
writes the metrics to `reports/benchmarks/deeploc.json`. You can
override the checkpoint or the benchmark directory:

```bash
uv run python -m src.baselines.deeploc_benchmark \
  --checkpoint models/checkpoints/best.ckpt \
  --benchmarks-dir benchmarks/deeploc
```

## Label mapping

DeepLoc location names are mapped to the project's internal classes
via the `DEFAULT_LABEL_MAP` dict in
`src/baselines/deeploc_benchmark.py`. Edit that dict if you change the
project taxonomy or if you add new DeepLoc locations.
