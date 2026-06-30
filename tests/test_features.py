"""Tests for motion feature extraction and dataset generation."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.features.dataset import (
    META_COLUMNS,
    FeatureDatasetBuilder,
    load_feature_dataset,
)
from src.features.extractor import (
    extract_window_features,
    extract_window_vector,
)
from src.optical_flow.dense_flow import DenseOpticalFlow
from src.pipeline.motion_pipeline import MotionPipeline
from src.utils.config import FeatureConfig, OpticalFlowConfig
from tests.synthetic import make_zone, write_synthetic_video


def _flow() -> DenseOpticalFlow:
    return DenseOpticalFlow(
        OpticalFlowConfig(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
    )


class FeatureExtractorTest(unittest.TestCase):
    """Validate individual window feature computations."""

    def test_basic_statistics(self) -> None:
        scores = [0.0, 1.0, 2.0, 3.0, 4.0]
        feats = extract_window_features(scores, threshold=1.0, feature_names=[
            "mean", "std", "max", "min", "median", "range",
        ])
        self.assertAlmostEqual(feats["mean"], 2.0)
        self.assertAlmostEqual(feats["max"], 4.0)
        self.assertAlmostEqual(feats["min"], 0.0)
        self.assertAlmostEqual(feats["median"], 2.0)
        self.assertAlmostEqual(feats["range"], 4.0)
        self.assertAlmostEqual(feats["std"], float(np.std(scores)))

    def test_active_ratio_and_mean_delta(self) -> None:
        feats = extract_window_features(
            [0.0, 1.0, 2.0], threshold=1.0, feature_names=["active_ratio"]
        )
        self.assertAlmostEqual(feats["active_ratio"], 2.0 / 3.0)

        feats = extract_window_features(
            [0.0, 2.0, 4.0], threshold=1.0, feature_names=["mean_delta"]
        )
        self.assertAlmostEqual(feats["mean_delta"], 2.0)

    def test_vector_preserves_requested_order(self) -> None:
        vector = extract_window_vector(
            [0.0, 1.0, 2.0, 3.0, 4.0], threshold=1.0, feature_names=["max", "mean"]
        )
        np.testing.assert_allclose(vector, [4.0, 2.0])

    def test_unknown_feature_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_window_features([1.0, 2.0], threshold=1.0, feature_names=["bogus"])

    def test_empty_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_window_features([], threshold=1.0, feature_names=["mean"])


class FeatureDatasetTest(unittest.TestCase):
    """Validate dataset generation and round-trip loading."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="features-test-"))
        self.video = self.temp_dir / "sample.mp4"
        write_synthetic_video(self.video, frames=60)
        self.feature_config = FeatureConfig(
            window_size=10, step=5, features=("mean", "std", "active_ratio")
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _builder(self) -> FeatureDatasetBuilder:
        pipeline = MotionPipeline(_flow(), [make_zone("COP"), make_zone("COK")])
        return FeatureDatasetBuilder(pipeline, self.feature_config)

    def test_write_csv_has_expected_columns_and_rows(self) -> None:
        csv_path = self.temp_dir / "features.csv"
        summary = self._builder().write_csv([self.video], csv_path)

        self.assertTrue(csv_path.exists())
        self.assertGreater(summary.total_samples, 0)
        self.assertEqual(set(summary.per_zone_counts), {"COP", "COK"})

        feature_names, matrices = load_feature_dataset(csv_path)
        self.assertEqual(feature_names, ["mean", "std", "active_ratio"])
        for zone, matrix in matrices.items():
            self.assertIn(zone, {"COP", "COK"})
            self.assertEqual(matrix.shape[1], 3)
            self.assertEqual(matrix.shape[0], summary.per_zone_counts[zone])

    def test_meta_columns_present_in_header(self) -> None:
        csv_path = self.temp_dir / "features.csv"
        self._builder().write_csv([self.video], csv_path)
        header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        for column in META_COLUMNS:
            self.assertIn(column, header)


if __name__ == "__main__":
    unittest.main()
