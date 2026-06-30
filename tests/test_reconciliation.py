"""Tests for the PLC + ERP + video reconciliation (Method 2 architecture)."""

from __future__ import annotations

import unittest

from src.erp.interface import MockErpClient, NullErpClient, ProductionOrder
from src.integration.reconciliation import (
    ProductionReconciler,
    ReconciliationStatus,
)
from src.plc.interface import MockPlcClient, NullPlcClient


def _order(zone: str = "COP") -> ProductionOrder:
    return ProductionOrder(
        zone=zone,
        order_id="WO-1",
        product_code="P-1",
        planned_start="2026-01-01T08:00:00",
        planned_end="2026-01-01T16:00:00",
    )


class ReconciliationTest(unittest.TestCase):
    """Validate the reconciliation decision matrix."""

    def test_no_order_is_planned_stop(self) -> None:
        reconciler = ProductionReconciler(MockPlcClient(), NullErpClient())
        result = reconciler.reconcile("COP", video_is_idle=True)
        self.assertEqual(result.status, ReconciliationStatus.PLANNED_STOP)
        self.assertFalse(result.alert)

    def test_running_and_idle_is_anomaly(self) -> None:
        reconciler = ProductionReconciler(
            MockPlcClient({"COP": True}), MockErpClient({"COP": _order()})
        )
        result = reconciler.reconcile("COP", video_is_idle=True)
        self.assertEqual(result.status, ReconciliationStatus.IDLE_WHILE_RUNNING)
        self.assertTrue(result.alert)

    def test_running_and_active_is_ok(self) -> None:
        reconciler = ProductionReconciler(
            MockPlcClient({"COP": True}), MockErpClient({"COP": _order()})
        )
        result = reconciler.reconcile("COP", video_is_idle=False)
        self.assertEqual(result.status, ReconciliationStatus.RUNNING)
        self.assertFalse(result.alert)

    def test_stopped_and_idle_is_unplanned_stop(self) -> None:
        reconciler = ProductionReconciler(
            MockPlcClient({"COP": False}), MockErpClient({"COP": _order()})
        )
        result = reconciler.reconcile("COP", video_is_idle=True)
        self.assertEqual(result.status, ReconciliationStatus.UNPLANNED_STOP)
        self.assertTrue(result.alert)

    def test_stopped_and_active_is_sensor_disagreement(self) -> None:
        reconciler = ProductionReconciler(
            MockPlcClient({"COP": False}), MockErpClient({"COP": _order()})
        )
        result = reconciler.reconcile("COP", video_is_idle=False)
        self.assertEqual(result.status, ReconciliationStatus.SENSOR_DISAGREEMENT)
        self.assertTrue(result.alert)

    def test_order_without_plc_falls_back_to_video(self) -> None:
        reconciler = ProductionReconciler(
            NullPlcClient(), MockErpClient({"COP": _order()})
        )
        idle = reconciler.reconcile("COP", video_is_idle=True)
        active = reconciler.reconcile("COP", video_is_idle=False)
        self.assertEqual(idle.status, ReconciliationStatus.IDLE_NO_PLC)
        self.assertTrue(idle.alert)
        self.assertEqual(active.status, ReconciliationStatus.RUNNING)
        self.assertFalse(active.alert)


if __name__ == "__main__":
    unittest.main()
