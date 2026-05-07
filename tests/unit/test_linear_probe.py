# tests/unit/test_linear_probe.py
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.baselines.linear_probe import (
    _discover_label_list,
    main,
    run_linear_probe,
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


class TestRunLinearProbe:
    @patch("src.baselines.linear_probe.compute_or_load_embeddings")
    @patch("src.baselines.linear_probe.compute_metrics")
    @patch("sklearn.multioutput.MultiOutputClassifier")
    @patch("sklearn.linear_model.LogisticRegression")
    def test_run_linear_probe_success(
        self, mock_lr, mock_moc, mock_compute_metrics, mock_embeddings, tmp_path
    ):
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
        mock_embeddings.side_effect = [
            (X, y, []),  # train
            (X, y, []),  # val
            (X, y, []),  # test
        ]

        # Mock classifier
        mock_clf_instance = MagicMock()
        # predict returns a list of arrays (one for each class) usually, but we mocked
        # predict to just return an array in the test here or just return a (10, 2)
        # array since it's converted to array.
        # Wait, the code says:
        # preds = clf.predict(features)
        # return np.asarray(preds, dtype=np.int64)
        mock_clf_instance.predict.return_value = np.zeros((10, 2))
        mock_moc.return_value = mock_clf_instance

        # Mock metrics
        mock_metrics_result = {
            "overall": {"f1_macro": 0.9},
            "per_class": {"ClassA": {"f1": 0.9}, "ClassB": {"f1": 0.9}},
        }
        mock_compute_metrics.return_value = mock_metrics_result

        summary = run_linear_probe(cfg, label_list, output_dir=tmp_path)

        assert summary["model"] == "linear_probe"
        assert summary["n_classes"] == 2
        assert summary["test"]["f1_macro"] == 0.9

        # Check that file was written
        report_file = tmp_path / "linear_probe.json"
        assert report_file.exists()

        with open(report_file) as f:
            data = json.load(f)
            assert data["model"] == "linear_probe"


class TestMain:
    @patch("src.baselines.linear_probe.run_linear_probe")
    @patch("src.baselines.linear_probe.load_config")
    @patch("src.baselines.linear_probe.setup_logging")
    @patch("src.baselines.linear_probe.argparse.ArgumentParser.parse_args")
    def test_main(
        self, mock_parse_args, mock_setup_logging, mock_load_config, mock_run_linear_probe
    ):
        mock_args = MagicMock()
        mock_args.max_iter = 500
        mock_args.regularization_strength = 2.0
        mock_args.overrides = []
        mock_parse_args.return_value = mock_args

        mock_cfg = MagicMock()
        mock_cfg.project.log_level = "INFO"
        mock_load_config.return_value = mock_cfg

        main()

        mock_load_config.assert_called_once_with(mode="training", overrides=[])
        mock_setup_logging.assert_called_once_with(level="INFO")
        mock_run_linear_probe.assert_called_once_with(
            mock_cfg, max_iter=500, regularization_strength=2.0
        )
