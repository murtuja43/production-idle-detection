"""Tests for the standalone evaluation utility."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import evaluate
from src.evaluation.ground_truth import (
    GroundTruthInterval,
    label_at,
    load_ground_truth,
    parse_label,
)
from src.evaluation.metrics import compute_metrics
from src.evaluation.runner import evaluate as run_eval
from src.evaluation.runner import evaluate_files, load_predictions


class MetricsTest(unittest.TestCase):
    """Validate metric computation."""

    def test_known_confusion(self) -> None:
        y_true = [True, True, False, False]
        y_pred = [True, False, False, False]
        m = compute_metrics(y_true, y_pred)
        self.assertEqual((m.tp, m.fp, m.fn, m.tn), (1, 0, 1, 2))
        self.assertAlmostEqual(m.accuracy, 0.75)
        self.assertAlmostEqual(m.precision, 1.0)
        self.assertAlmostEqual(m.recall, 0.5)
        self.assertAlmostEqual(m.f1, 2 / 3)

    def test_empty_is_safe(self) -> None:
        m = compute_metrics([], [])
        self.assertEqual(m.support, 0)
        self.assertEqual(m.f1, 0.0)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics([True], [True, False])


class GroundTruthTest(unittest.TestCase):
    """Validate ground-truth parsing and lookup."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gt-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_parse_label_variants(self) -> None:
        self.assertTrue(parse_label("idle"))
        self.assertTrue(parse_label("1"))
        self.assertTrue(parse_label("TRUE"))
        self.assertFalse(parse_label("active"))
        self.assertFalse(parse_label("0"))
        with self.assertRaises(ValueError):
            parse_label("maybe")

    def test_load_flexible_columns(self) -> None:
        path = self.temp_dir / "gt.csv"
        path.write_text(
            "zone,start_time,end_time,state\n"
            "COP,0.0,2.0,idle\n"
            "COP,2.0,4.0,active\n",
            encoding="utf-8",
        )
        intervals = load_ground_truth(path)
        self.assertEqual(len(intervals), 2)
        self.assertTrue(intervals[0].is_idle)
        self.assertFalse(intervals[1].is_idle)

    def test_label_at_precedence_and_default(self) -> None:
        intervals = [
            GroundTruthInterval("COP", 0.0, 2.0, True),
            GroundTruthInterval(None, 0.0, 2.0, False),  # global, overlaps
        ]
        # idle takes precedence among overlapping intervals
        self.assertTrue(label_at(intervals, "COP", 1.0, default=False))
        # global interval applies to other zones
        self.assertFalse(label_at(intervals, "COK", 1.0, default=True))
        # outside all intervals -> default
        self.assertTrue(label_at(intervals, "COP", 5.0, default=True))
        self.assertIsNone(label_at(intervals, "COP", 5.0, default=None))

    def test_bad_interval_raises(self) -> None:
        path = self.temp_dir / "bad.csv"
        path.write_text("zone,start,end,label\nCOP,4.0,1.0,idle\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_ground_truth(path)


class EvaluateTest(unittest.TestCase):
    """Validate alignment, evaluation, and output writing."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="eval-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _write_predictions(self) -> Path:
        path = self.temp_dir / "pred.csv"
        path.write_text(
            "frame_index,timestamp_seconds,zone,motion_score,motion_threshold,"
            "is_motion_active,is_idle,idle_seconds,sensitivity\n"
            "1,0.0,COP,0.0,1.0,0,1,0.0,1.0\n"   # pred idle  @0s
            "2,1.0,COP,0.0,1.0,0,1,0.0,1.0\n"   # pred idle  @1s
            "3,2.0,COP,2.0,1.0,1,0,0.0,1.0\n",  # pred active@2s
            encoding="utf-8",
        )
        return path

    def _write_ground_truth(self) -> Path:
        path = self.temp_dir / "gt.csv"
        path.write_text(
            "zone,start_seconds,end_seconds,label\n"
            "COP,0.0,1.5,idle\n"      # 0s,1s -> idle
            "COP,1.5,3.0,active\n",   # 2s   -> active
            encoding="utf-8",
        )
        return path

    def test_load_predictions(self) -> None:
        by_zone = load_predictions(self._write_predictions())
        self.assertEqual(set(by_zone), {"COP"})
        self.assertEqual(len(by_zone["COP"]), 3)
        self.assertEqual(by_zone["COP"][0], (0.0, True))

    def test_perfect_evaluation(self) -> None:
        result = evaluate_files(
            self._write_predictions(), self._write_ground_truth(),
            default_label="active",
        )
        # GT: idle,idle,active ; Pred: idle,idle,active -> perfect
        self.assertEqual(result.overall.tp, 2)
        self.assertEqual(result.overall.tn, 1)
        self.assertAlmostEqual(result.overall.f1, 1.0)
        self.assertAlmostEqual(result.overall.accuracy, 1.0)

    def test_default_skip_excludes_uncovered(self) -> None:
        # GT only covers 0-1.5s; the 2s frame is uncovered.
        gt = self.temp_dir / "gt_partial.csv"
        gt.write_text(
            "zone,start_seconds,end_seconds,label\nCOP,0.0,1.5,idle\n",
            encoding="utf-8",
        )
        predictions = load_predictions(self._write_predictions())
        intervals = load_ground_truth(gt)
        result = run_eval(predictions, intervals, default_label="skip")
        self.assertEqual(result.skipped_frames, 1)
        self.assertEqual(result.overall.support, 2)

    def test_zone_filter(self) -> None:
        by_zone = load_predictions(self._write_predictions(), zone_filter="COK")
        self.assertEqual(by_zone, {})

    def test_cli_writes_outputs(self) -> None:
        out_dir = self.temp_dir / "out"
        args = argparse.Namespace(
            predictions=str(self._write_predictions()),
            ground_truth=str(self._write_ground_truth()),
            output_dir=str(out_dir),
            default_label="active",
            zone=None,
            metrics_json="m.json",
            metrics_csv="m.csv",
            plot="cmp.png",
            no_plot=False,
        )
        result = evaluate.run_evaluation(args)
        self.assertTrue((out_dir / "m.json").exists())
        self.assertTrue((out_dir / "m.csv").exists())
        self.assertTrue((out_dir / "cmp.png").exists())
        data = json.loads((out_dir / "m.json").read_text(encoding="utf-8"))
        self.assertIn("overall", data)
        self.assertEqual(data["overall"]["f1"], result.overall.f1)


if __name__ == "__main__":
    unittest.main()
