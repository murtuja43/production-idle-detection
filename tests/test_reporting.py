"""Tests for per-zone reporting (aggregation + CSV/JSON/chart output)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.reporting.report import ReportAggregator


class ReportAggregatorTest(unittest.TestCase):
    """Validate statistics aggregation and report serialization."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="report-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_counts_idle_events_and_times(self) -> None:
        agg = ReportAggregator(zone_names=["A"], mode="optical_flow", fps=10.0)
        # active, idle, idle, active, idle -> 2 idle episodes, 3 idle frames.
        pattern = [
            (1.0, False), (0.0, True), (0.0, True), (2.0, False), (0.0, True),
        ]
        for motion, is_idle in pattern:
            agg.update("A", motion, is_idle)

        report = agg.build(video="v.mp4")
        zone = report.zones[0]
        self.assertEqual(zone.frames, 5)
        self.assertEqual(zone.idle_events, 2)
        self.assertAlmostEqual(zone.idle_seconds, 0.3)
        self.assertAlmostEqual(zone.active_seconds, 0.2)
        self.assertAlmostEqual(zone.total_seconds, 0.5)
        self.assertAlmostEqual(zone.average_motion, 0.6)

    def test_anomaly_count(self) -> None:
        agg = ReportAggregator(zone_names=["A"], mode="ml", fps=10.0)
        agg.update("A", 1.0, False, is_anomaly=True)
        agg.update("A", 1.0, True, is_anomaly=True)
        agg.update("A", 1.0, False, is_anomaly=False)
        report = agg.build(video="v.mp4")
        self.assertEqual(report.zones[0].anomaly_count, 2)
        self.assertEqual(report.summary()["total_anomalies"], 2)

    def test_save_csv_and_json(self) -> None:
        agg = ReportAggregator(zone_names=["A", "B"], mode="combined", fps=25.0)
        for _ in range(5):
            agg.update("A", 0.0, True)
            agg.update("B", 3.0, False)
        report = agg.build(video="clip.mp4")

        csv_path = self.temp_dir / "report.csv"
        json_path = self.temp_dir / "report.json"
        report.save_csv(csv_path)
        report.save_json(json_path)

        self.assertTrue(csv_path.exists())
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["summary"]["mode"], "combined")
        self.assertEqual(data["summary"]["video"], "clip.mp4")
        self.assertEqual(len(data["zones"]), 2)

    def test_save_chart_creates_file(self) -> None:
        agg = ReportAggregator(zone_names=["A"], mode="optical_flow", fps=10.0)
        agg.update("A", 0.0, True)
        report = agg.build(video="v.mp4")
        chart_path = self.temp_dir / "report.png"
        produced = report.save_chart(chart_path)
        # matplotlib is a declared dependency, so the chart should be produced.
        self.assertTrue(produced)
        self.assertTrue(chart_path.exists())


if __name__ == "__main__":
    unittest.main()
