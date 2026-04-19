# tests/unit/test_deeploc_benchmark.py
"""Tests for DeepLoc benchmark input loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.baselines.deeploc_benchmark import DEFAULT_LABEL_MAP, _load_deeploc_test_set


def _write_demo_fasta(path: Path) -> None:
    path.write_text(
        ">sp|P07221|CASQ1_RABIT Example one\n"
        "MNAADRMGARVALLLLL\n"
        ">sp|P82413|RK19_SPIOL Example two\n"
        "MASKVLPQALLVIPSN\n"
        ">sp|P30952|MLS1_YEAST Example three\n"
        "MVKVSLDNVKLLVDVD\n"
    )


class TestLoadDeepLocReferenceSet:
    """Tests for DeepLoc benchmark file auto-detection."""

    def test_supports_deeploc_package_results_csv(self, tmp_path: Path) -> None:
        bench_dir = tmp_path / "deeploc2_package"
        outputs_dir = bench_dir / "outputs"
        outputs_dir.mkdir(parents=True)

        _write_demo_fasta(bench_dir / "test.fasta")
        (outputs_dir / "results_test.csv").write_text(
            "\n".join(
                [
                    (
                        "Protein_ID,Localizations,Signals,Cytoplasm,Nucleus,Extracellular,"
                        "Cell membrane,Mitochondrion,Plastid,Endoplasmic reticulum,"
                        "Lysosome/Vacuole,Golgi apparatus,Peroxisome"
                    ),
                    (
                        "sp|P07221|CASQ1_RABIT,Extracellular|Endoplasmic reticulum,"
                        "Signal peptide,0,0,0,0,0,0,0,0,0,0"
                    ),
                    "sp|P82413|RK19_SPIOL,Plastid,Transit peptide,0,0,0,0,0,0,0,0,0,0",
                    "sp|P30952|MLS1_YEAST,Peroxisome,PTS1,0,0,0,0,0,0,0,0,0,0",
                ]
            )
        )

        df, metadata = _load_deeploc_test_set(bench_dir, DEFAULT_LABEL_MAP)

        assert metadata["reference_type"] == "deeploc_predictions"
        assert metadata["n_source_rows"] == 3
        assert metadata["n_skipped_unmapped_labels"] == 1
        assert metadata["unmapped_labels"] == ["Plastid"]
        assert df["accession"].tolist() == [
            "sp|P07221|CASQ1_RABIT",
            "sp|P30952|MLS1_YEAST",
        ]
        assert df["locations_str"].tolist() == [
            "Secreted/Extracellular|Endoplasmic Reticulum",
            "Peroxisome",
        ]

    def test_supports_accession_only_ground_truth_tsv(self, tmp_path: Path) -> None:
        bench_dir = tmp_path / "deeploc"
        bench_dir.mkdir()

        _write_demo_fasta(bench_dir / "test.fasta")
        (bench_dir / "test_labels.tsv").write_text(
            "P07221\tExtracellular|Endoplasmic reticulum\nP30952\tPeroxisome\n"
        )

        df, metadata = _load_deeploc_test_set(bench_dir, DEFAULT_LABEL_MAP)

        assert metadata["reference_type"] == "ground_truth_labels"
        assert metadata["n_source_rows"] == 2
        assert df["accession"].tolist() == [
            "sp|P07221|CASQ1_RABIT",
            "sp|P30952|MLS1_YEAST",
        ]

    def test_invalid_tsv_raises_helpful_error_without_results_csv(self, tmp_path: Path) -> None:
        bench_dir = tmp_path / "deeploc"
        bench_dir.mkdir()

        _write_demo_fasta(bench_dir / "test.fasta")
        (bench_dir / "test_labels.tsv").write_text(
            "sp|P07221|CASQ1_RABIT\tCalsequestrin-1\nsp|P82413|RK19_SPIOL\t50S\n"
        )

        with pytest.raises(RuntimeError, match="did not contain recognizable location labels"):
            _load_deeploc_test_set(bench_dir, DEFAULT_LABEL_MAP)
