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

