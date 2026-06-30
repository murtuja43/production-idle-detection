"""Reconcile video idle detection with PLC state and the ERP schedule.

This module is the architectural blueprint for **Method 2**. It is intentionally
NOT wired into the main pipeline yet: it depends only on the abstract
:class:`~src.plc.interface.PlcClient` and :class:`~src.erp.interface.ErpClient`
contracts, so real PLC/ERP clients can be dropped in later without touching the
reconciliation logic.

Decision matrix (per zone), given the video idle verdict from Method 1:

| ERP order | PLC running | Video idle | Result                         | Alert |
|-----------|-------------|------------|--------------------------------|-------|
| none      | any         | any        | PLANNED_STOP                   | no    |
| present   | running     | active     | RUNNING                        | no    |
| present   | running     | idle       | IDLE_WHILE_RUNNING (anomaly!)  | yes   |
| present   | stopped     | idle       | UNPLANNED_STOP                 | yes   |
| present   | stopped     | active     | SENSOR_DISAGREEMENT            | yes   |
| present   | unknown     | idle       | IDLE_NO_PLC                    | yes   |
| present   | unknown     | active     | RUNNING                        | no    |
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.erp.interface import ErpClient
from src.plc.interface import PlcClient


class ReconciliationStatus(str, Enum):
    """Outcome of reconciling the three signals for a zone."""

    PLANNED_STOP = "planned_stop"
    RUNNING = "running"
    IDLE_WHILE_RUNNING = "idle_while_running"
    UNPLANNED_STOP = "unplanned_stop"
    SENSOR_DISAGREEMENT = "sensor_disagreement"
    IDLE_NO_PLC = "idle_no_plc"


@dataclass(frozen=True)
class ReconciliationResult:
    """Reconciled verdict for one zone."""

    zone: str
    status: ReconciliationStatus
    alert: bool
    detail: str


class ProductionReconciler:
    """Combine PLC state + ERP schedule + video analysis into a verdict.

    The video idle verdict (``video_is_idle``) comes from Method 1; the PLC and
    ERP signals come from the injected clients. The default (Null) clients yield
    ``IDLE_NO_PLC`` / video-driven behaviour, which is exactly the MVP situation.
    """

    def __init__(self, plc_client: PlcClient, erp_client: ErpClient) -> None:
        self._plc = plc_client
        self._erp = erp_client

    def reconcile(self, zone: str, video_is_idle: bool) -> ReconciliationResult:
        """Reconcile the three signals for a single zone."""
        order = self._erp.current_order(zone)
        if order is None:
            return ReconciliationResult(
                zone=zone,
                status=ReconciliationStatus.PLANNED_STOP,
                alert=False,
                detail="No scheduled production; idle time is expected.",
            )

        signal = self._plc.read_signal(zone)
        plc_running = signal.is_running if signal is not None else None

        if plc_running is None:
            if video_is_idle:
                return ReconciliationResult(
                    zone, ReconciliationStatus.IDLE_NO_PLC, True,
                    "Production scheduled and video idle, but no PLC confirmation.",
                )
            return ReconciliationResult(
                zone, ReconciliationStatus.RUNNING, False,
                "Production scheduled and video shows activity.",
            )

        if plc_running and video_is_idle:
            return ReconciliationResult(
                zone, ReconciliationStatus.IDLE_WHILE_RUNNING, True,
                "Conveyor running but video shows no activity (likely stoppage).",
            )
        if plc_running and not video_is_idle:
            return ReconciliationResult(
                zone, ReconciliationStatus.RUNNING, False,
                "Conveyor running and video shows activity.",
            )
        if not plc_running and video_is_idle:
            return ReconciliationResult(
                zone, ReconciliationStatus.UNPLANNED_STOP, True,
                "Production scheduled but conveyor stopped and video idle.",
            )
        return ReconciliationResult(
            zone, ReconciliationStatus.SENSOR_DISAGREEMENT, True,
            "PLC reports stopped but video shows activity; check sensors.",
        )
