# DeepLoc 2.0 benchmark

This directory holds the DeepLoc 2.0 test set used by
`src.baselines.deeploc_benchmark`. The data is **not** included in this
repository because the upstream dataset has its own license terms.

## What to put here

You can use either of these layouts:

1. True benchmark labels:

- `test.fasta` — protein sequences in FASTA format
- `test_labels.tsv` — tab-separated, two columns: accession and the
  pipe-separated list of subcellular locations

Example `test_labels.tsv` row:

```
P12345 Nucleus|Cytoplasm
```

The script matches both plain accessions like `P12345` and full FASTA
ids like `sp|P12345|NAME_HUMAN`.

2. Packaged DeepLoc 2.0 demo files:

- `test.fasta`
- `outputs/results_test.csv`

In the second case, the benchmark compares this project's predictions
against DeepLoc's own packaged predictions. That is useful as a
compatibility/agreement check, but it is **not** a ground-truth
benchmark.

## How to obtain the dataset

1. Visit <https://services.healthtech.dtu.dk/services/DeepLoc-2.0/>
2. Request access to the dataset (academic use is typically free).
3. Download the test set distribution.
4. Place the FASTA and label files into this directory with the names
   above, or point `--benchmarks-dir` at the unpacked `deeploc2_package`
   directory.

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

Example using the packaged DeepLoc 2.0 files directly:

```bash
uv run python -m src.baselines.deeploc_benchmark \
  --checkpoint models/checkpoints/best.ckpt \
  --benchmarks-dir /path/to/deeploc2_package
```

## Label mapping

DeepLoc location names are mapped to the project's internal classes
via the `DEFAULT_LABEL_MAP` dict in
`src/baselines/deeploc_benchmark.py`. Edit that dict if you change the
project taxonomy or if you add new DeepLoc locations.
