"""PLC integration contract for future production signals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PlcSignal:
    """A single PLC signal sample."""

    zone: str
    timestamp_seconds: float
    is_running: bool
    raw_value: float | int | bool


class PlcClient(ABC):
    """Interface for future PLC integrations."""

    @abstractmethod
    def read_signal(self, zone: str) -> PlcSignal | None:
        """Read the latest PLC signal for a production zone."""


class NullPlcClient(PlcClient):
    """No-op PLC client used by the video-only MVP."""

    def read_signal(self, zone: str) -> PlcSignal | None:
        """Return no PLC signal because PLC integration is not enabled."""
        return None


class MockPlcClient(PlcClient):
    """In-memory PLC client for demos and tests (Method 2 placeholder).

    Real deployments would replace this with a client speaking Modbus/OPC-UA/etc.
    while keeping the :class:`PlcClient` contract unchanged.
    """

    def __init__(
        self,
        running_by_zone: dict[str, bool] | None = None,
        default_running: bool = True,
    ) -> None:
        self._running_by_zone = dict(running_by_zone or {})
        self._default_running = default_running

    def set_running(self, zone: str, is_running: bool) -> None:
        """Update the simulated conveyor state for a zone."""
        self._running_by_zone[zone] = is_running

    def read_signal(self, zone: str) -> PlcSignal | None:
        """Return the simulated running state for a zone."""
        is_running = self._running_by_zone.get(zone, self._default_running)
        return PlcSignal(
            zone=zone,
            timestamp_seconds=0.0,
            is_running=is_running,
            raw_value=bool(is_running),
        )

