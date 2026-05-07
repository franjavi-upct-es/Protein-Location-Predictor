# tests/unit/test_download.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.download import (
    _download_organism,
    _get_next_link,
    _normalize_columns,
    download_proteins,
    main,
)
from src.utils.config import DotDict


class TestGetNextLink:
    def test_valid_link(self):
        headers = {"Link": '<https://api.uniprot.org/next>; rel="next"'}
        assert _get_next_link(headers) == "https://api.uniprot.org/next"

    def test_no_link(self):
        assert _get_next_link({}) is None

    def test_invalid_link_format(self):
        headers = {"Link": "something else"}
        assert _get_next_link(headers) is None


class TestDownloadOrganism:
    def test_single_page(self):
        session = MagicMock()
        response = MagicMock()
        response.text = "Entry\tSequence\n1\tA\n2\tB"
        response.headers = {}
        session.get.return_value = response

        df = _download_organism(session, "http://api", 9606, True, "Entry,Sequence")

        assert len(df) == 2
        assert "Entry" in df.columns

    def test_pagination(self):
        session = MagicMock()

        resp1 = MagicMock()
        resp1.text = "Entry\tSequence\n1\tA"
        resp1.headers = {"Link": '<http://next>; rel="next"'}

        resp2 = MagicMock()
        resp2.text = "Entry\tSequence\n2\tB"
        resp2.headers = {}

        session.get.side_effect = [resp1, resp2]

        df = _download_organism(session, "http://api", 9606, True, "Entry,Sequence")

        assert len(df) == 2
        assert session.get.call_count == 2

    def test_max_results_limit_hit(self):
        session = MagicMock()
        response = MagicMock()
        response.text = "Entry\tSequence\n1\tA\n2\tB\n3\tC"
        response.headers = {"Link": '<http://next>; rel="next"'}
        session.get.return_value = response

        df = _download_organism(session, "http://api", 9606, True, "Entry,Sequence", max_results=2)

        assert len(df) == 2
        assert session.get.call_count == 1

    def test_empty_response(self):
        session = MagicMock()
        response = MagicMock()
        response.text = "Entry\tSequence\n"
        response.headers = {}
        session.get.return_value = response

        df = _download_organism(session, "http://api", 9606, True, "Entry,Sequence")

        assert len(df) == 0


class TestNormalizeColumns:
    def test_valid_columns(self):
        df = pd.DataFrame(
            {
                "Entry": ["A"],
                "Sequence": ["M"],
                "Subcellular location [CC]": ["Loc"],
            }
        )

        normalized = _normalize_columns(df)
        assert "accession" in normalized.columns
        assert "sequence" in normalized.columns
        assert "subcellular_location" in normalized.columns

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"Entry": ["A"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            _normalize_columns(df)


class TestDownloadProteins:
    def test_success_with_deduplication(self, monkeypatch, tmp_path):
        cfg = DotDict.from_dict(
            {
                "project_root": str(tmp_path),
                "data": {
                    "uniprot_api_url": "http://api",
                    "fields": "Entry",
                    "sources": [{"organism_id": 9606, "reviewed": True}],
                },
                "paths": {"raw_dir": "raw_dir"},
            }
        )

        mock_df = pd.DataFrame(
            {
                "Entry": ["A", "B", "A"],
                "Sequence": ["M", "M", "M"],
                "Subcellular location [CC]": ["Loc", "Loc", "Loc"],
            }
        )

        monkeypatch.setattr("src.data.download._download_organism", lambda **kwargs: mock_df)

        df = download_proteins(cfg)

        assert len(df) == 2
        assert list(df["accession"]) == ["A", "B"]
        assert (tmp_path / "raw_dir" / "uniprot_data.csv").exists()

    def test_no_data_raises(self, monkeypatch, tmp_path):
        cfg = DotDict.from_dict(
            {"data": {"uniprot_api_url": "http://api", "sources": [{"organism_id": 9606}]}}
        )

        monkeypatch.setattr("src.data.download._download_organism", lambda **kwargs: pd.DataFrame())

        with pytest.raises(RuntimeError, match="No data downloaded"):
            download_proteins(cfg)


class TestMain:
    @patch("src.data.download.download_proteins")
    @patch("src.data.download.load_config")
    @patch("src.data.download.setup_logging")
    @patch("src.data.download.argparse.ArgumentParser.parse_args")
    def test_main(
        self, mock_parse_args, mock_setup_logging, mock_load_config, mock_download_proteins
    ):
        mock_args = MagicMock()
        mock_args.max_results = 100
        mock_args.config_overrides = []
        mock_parse_args.return_value = mock_args

        mock_cfg = MagicMock()
        mock_cfg.project.log_level = "INFO"
        mock_load_config.return_value = mock_cfg

        main()

        mock_load_config.assert_called_once_with(overrides=[])
        mock_setup_logging.assert_called_once_with(level="INFO")
        mock_download_proteins.assert_called_once_with(mock_cfg, max_results_per_organism=100)

    @patch("src.data.download.download_proteins")
    @patch("src.data.download.load_config")
    @patch("src.data.download.setup_logging")
    @patch("src.data.download.argparse.ArgumentParser.parse_args")
    def test_main_unlimited_results(
        self, mock_parse_args, mock_setup_logging, mock_load_config, mock_download_proteins
    ):
        mock_args = MagicMock()
        mock_args.max_results = -1
        mock_args.config_overrides = []
        mock_parse_args.return_value = mock_args

        main()

        mock_download_proteins.assert_called_once()
        assert mock_download_proteins.call_args[1]["max_results_per_organism"] is None
