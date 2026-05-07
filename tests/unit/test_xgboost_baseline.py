# tests/unit/test_xgboost_baseline.py
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.baselines.xgboost_baseline import (
    _discover_label_list,
    main,
    run_xgboost_baseline,
)
from src.utils.config import DotDict


class TestDiscoverLabelList:
    def test_discover_label_list(self, tmp_path, monkeypatch):
        cfg = DotDict.from_dict(
            {"project_root": str(tmp_path), "paths": {"splits_dir": str(tmp_path)}}
        )

        train_df = pd.DataFrame({"locations_str": ["Cytoplasm|Nucleus", "Membrane", "Cytoplasm"]})
        train_df.to_csv(tmp_path / "train.csv", index=False)

        labels = _discover_label_list(cfg)
        assert labels == ["Cytoplasm", "Membrane", "Nucleus"]


class TestRunXgboostBaseline:
    @patch("src.baselines.xgboost_baseline.compute_or_load_embeddings")
    @patch("src.baselines.xgboost_baseline.compute_metrics")
    def test_run_xgboost_baseline_success(
        self, mock_compute_metrics, mock_embeddings, tmp_path, monkeypatch
    ):
        # Mock xgboost module
        mock_xgb = MagicMock()
        mock_clf_class = MagicMock()

        # Instance returned by XGBClassifier
        mock_clf_instance = MagicMock()
        mock_clf_instance.predict.return_value = np.zeros(10)
        mock_clf_class.return_value = mock_clf_instance

        mock_xgb.XGBClassifier = mock_clf_class
        monkeypatch.setitem(sys.modules, "xgboost", mock_xgb)

        cfg = DotDict.from_dict(
            {
                "project_root": str(tmp_path),
                "model": {"backbone": {"name": "test_backbone"}},
                "paths": {"reports_dir": str(tmp_path)},
            }
        )

        label_list = ["ClassA", "ClassB"]

        # Mock embeddings
        X = np.zeros((10, 5))
        y = np.zeros((10, 2))
        y[0, 0] = 1  # One positive for ClassA to test scale_pos_weight
        mock_embeddings.side_effect = [
            (X, y, []),  # train
            (X, y, []),  # val
            (X, y, []),  # test
        ]

        # Mock metrics
        mock_metrics_result = {
            "overall": {"f1_macro": 0.9},
            "per_class": {"ClassA": {"f1": 0.9}, "ClassB": {"f1": 0.9}},
        }
        mock_compute_metrics.return_value = mock_metrics_result

        summary = run_xgboost_baseline(cfg, label_list, output_dir=tmp_path)

        assert summary["model"] == "xgboost_baseline"
        assert summary["n_classes"] == 2
        assert summary["test"]["f1_macro"] == 0.9

        # Check that file was written
        report_file = tmp_path / "xgboost_baseline.json"
        assert report_file.exists()

        with open(report_file) as f:
            data = json.load(f)
            assert data["model"] == "xgboost_baseline"

    def test_run_xgboost_import_error(self, tmp_path, monkeypatch):
        # Remove xgboost from sys.modules to simulate ImportError
        if "xgboost" in sys.modules:
            monkeypatch.delitem(sys.modules, "xgboost")

        # We need to explicitly cause the import inside the function to fail
        # because the module might be cached or lazily loaded. We patch importlib
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "xgboost":
                raise ImportError("No module named xgboost")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        cfg = DotDict.from_dict({})
        with pytest.raises(ImportError, match="XGBoost is not installed"):
            run_xgboost_baseline(cfg)


class TestMain:
    @patch("src.baselines.xgboost_baseline.run_xgboost_baseline")
    @patch("src.baselines.xgboost_baseline.load_config")
    @patch("src.baselines.xgboost_baseline.setup_logging")
    @patch("src.baselines.xgboost_baseline.argparse.ArgumentParser.parse_args")
    def test_main(
        self, mock_parse_args, mock_setup_logging, mock_load_config, mock_run_xgboost_baseline
    ):
        mock_args = MagicMock()
        mock_args.n_estimators = 500
        mock_args.max_depth = 4
        mock_args.learning_rate = 0.05
        mock_args.overrides = []
        mock_parse_args.return_value = mock_args

        mock_cfg = MagicMock()
        mock_cfg.project.log_level = "INFO"
        mock_load_config.return_value = mock_cfg

        main()

        mock_load_config.assert_called_once_with(mode="training", overrides=[])
        mock_setup_logging.assert_called_once_with(level="INFO")
        mock_run_xgboost_baseline.assert_called_once_with(
            mock_cfg, n_estimators=500, max_depth=4, learning_rate=0.05
        )
