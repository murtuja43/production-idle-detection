"""Tests for Isolation Forest training and model persistence."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from src.features.dataset import FeatureDatasetBuilder
from src.ml.model import IdleAnomalyModel, ModelMetadata
from src.optical_flow.dense_flow import DenseOpticalFlow
from src.pipeline.motion_pipeline import MotionPipeline
from src.training.trainer import train_from_feature_csv
from src.utils.config import FeatureConfig, OpticalFlowConfig, TrainingConfig
from tests.synthetic import make_zone, write_synthetic_video


def _flow() -> DenseOpticalFlow:
    return DenseOpticalFlow(
        OpticalFlowConfig(0.5, 3, 15, 3, 5, 1.2, 0)
    )


def _metadata(feature_names: list[str]) -> ModelMetadata:
    return ModelMetadata(
        feature_names=feature_names,
        window_size=4,
        step=2,
        contamination="auto",
        random_state=0,
        n_estimators=50,
        max_samples="auto",
        zones=["COP"],
        zone_sample_counts={"COP": 32},
        sklearn_version="test",
        created_at="2026-01-01T00:00:00+00:00",
        source="unit-test",
    )


class ModelPersistenceTest(unittest.TestCase):
    """Validate model + metadata save/load round-trips."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="model-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_metadata_dict_round_trip(self) -> None:
        meta = _metadata(["mean", "std"])
        restored = ModelMetadata.from_dict(meta.to_dict())
        self.assertEqual(restored, meta)

    def test_model_save_load_predict(self) -> None:
        features = ["mean", "std"]
        rng = np.random.RandomState(0)
        matrix = rng.rand(40, len(features))
        estimator = IsolationForest(random_state=0).fit(matrix)
        model = IdleAnomalyModel({"COP": estimator}, _metadata(features))

        model_path = self.temp_dir / "m.joblib"
        meta_path = self.temp_dir / "m.json"
        model.save(model_path, meta_path)
        self.assertTrue(model_path.exists())
        self.assertTrue(meta_path.exists())

        loaded = IdleAnomalyModel.load(model_path, meta_path)
        self.assertEqual(loaded.feature_names, ("mean", "std"))
        self.assertTrue(loaded.has_zone("COP"))
        is_anomaly, score = loaded.predict("COP", matrix[0])
        self.assertIsInstance(is_anomaly, bool)
        self.assertIsInstance(score, float)

    def test_predict_wrong_feature_count_raises(self) -> None:
        estimator = IsolationForest(random_state=0).fit(np.random.rand(20, 2))
        model = IdleAnomalyModel({"COP": estimator}, _metadata(["mean", "std"]))
        with self.assertRaises(ValueError):
            model.predict("COP", np.array([1.0, 2.0, 3.0]))

    def test_predict_unknown_zone_raises(self) -> None:
        estimator = IsolationForest(random_state=0).fit(np.random.rand(20, 2))
        model = IdleAnomalyModel({"COP": estimator}, _metadata(["mean", "std"]))
        with self.assertRaises(KeyError):
            model.predict("NOPE", np.array([1.0, 2.0]))


class TrainerTest(unittest.TestCase):
    """Validate end-to-end training from generated features."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="trainer-test-"))
        self.video = self.temp_dir / "sample.mp4"
        write_synthetic_video(self.video, frames=80)
        self.feature_config = FeatureConfig(
            window_size=10, step=5, features=("mean", "std", "max", "active_ratio")
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _make_feature_csv(self) -> Path:
        pipeline = MotionPipeline(_flow(), [make_zone("COP"), make_zone("CMUS")])
        builder = FeatureDatasetBuilder(pipeline, self.feature_config)
        csv_path = self.temp_dir / "features.csv"
        builder.write_csv([self.video], csv_path)
        return csv_path

    def test_train_produces_model_and_metadata(self) -> None:
        csv_path = self._make_feature_csv()
        training_config = TrainingConfig(
            contamination="auto",
            random_state=42,
            n_estimators=50,
            max_samples="auto",
            feature_csv=str(csv_path),
            model_output=str(self.temp_dir / "model.joblib"),
            metadata_output=str(self.temp_dir / "model.json"),
        )
        result = train_from_feature_csv(
            csv_path, training_config, self.feature_config, source="unit-test"
        )

        self.assertTrue(result.model_path.exists())
        self.assertTrue(result.metadata_path.exists())
        self.assertEqual(set(result.zone_sample_counts), {"COP", "CMUS"})

        loaded = IdleAnomalyModel.load(result.model_path, result.metadata_path)
        self.assertEqual(
            loaded.feature_names, ("mean", "std", "max", "active_ratio")
        )
        self.assertEqual(loaded.window_size, 10)
        self.assertEqual(set(loaded.zones()), {"COP", "CMUS"})
        self.assertEqual(loaded.metadata.source, "unit-test")

    def test_empty_csv_raises(self) -> None:
        empty = self.temp_dir / "empty.csv"
        empty.write_text("source,zone,window_index,mean\n", encoding="utf-8")
        training_config = TrainingConfig(
            "auto", 42, 50, "auto",
            str(empty), str(self.temp_dir / "m.joblib"), str(self.temp_dir / "m.json"),
        )
        with self.assertRaises(ValueError):
            train_from_feature_csv(empty, training_config, self.feature_config)


if __name__ == "__main__":
    unittest.main()
