"""Stateful idle detection from per-zone motion scores."""

from __future__ import annotations

from dataclasses import dataclass

from src.preprocessing.roi import Zone, ZoneRegistry


@dataclass(frozen=True)
class ZoneDetectionState:
    """Current idle/active state for a zone."""

    zone_name: str
    is_idle: bool
    is_motion_active: bool
    idle_seconds: float
    motion_score: float
    threshold: float
    timestamp_seconds: float


@dataclass
class _ZoneRuntimeState:
    zone: Zone
    idle_started_at: float | None = None
    latest_state: ZoneDetectionState | None = None


class IdleDetector:
    """Detect idle periods when motion remains below threshold long enough."""

    def __init__(self, zones: list[Zone]) -> None:
        self._states = {
            zone.name: _ZoneRuntimeState(zone=zone)
            for zone in zones
            if zone.enabled
        }

    @classmethod
    def from_zones(cls, registry: ZoneRegistry) -> "IdleDetector":
        """Create detector from enabled zones in a registry."""
        return cls(registry.enabled_zones)

    def update(
        self,
        zone_name: str,
        motion_score: float,
        timestamp_seconds: float,
    ) -> ZoneDetectionState:
        """Update one zone with a motion score and return its state."""
        if zone_name not in self._states:
            raise KeyError(f"Unknown or disabled zone: {zone_name}")

        runtime = self._states[zone_name]
        zone = runtime.zone
        is_motion_active = motion_score >= zone.motion_threshold

        if is_motion_active:
            runtime.idle_started_at = None
            idle_seconds = 0.0
            is_idle = False
        else:
            if runtime.idle_started_at is None:
                runtime.idle_started_at = timestamp_seconds
            idle_seconds = timestamp_seconds - runtime.idle_started_at
            is_idle = idle_seconds >= zone.idle_duration_seconds

        state = ZoneDetectionState(
            zone_name=zone.name,
            is_idle=is_idle,
            is_motion_active=is_motion_active,
            idle_seconds=max(0.0, idle_seconds),
            motion_score=motion_score,
            threshold=zone.motion_threshold,
            timestamp_seconds=timestamp_seconds,
        )
        runtime.latest_state = state
        return state

    def latest_states(self) -> list[ZoneDetectionState]:
        """Return latest known states for all zones."""
        return [
            runtime.latest_state
            for runtime in self._states.values()
            if runtime.latest_state is not None
        ]

