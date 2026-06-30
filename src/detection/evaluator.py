"""Per-frame, per-zone idle evaluation across all detection modes.

The optical-flow detector ALWAYS runs (preserving Phase 1 behavior exactly); the
ML classifier is additive. The selected mode only decides how the final idle
decision is derived:

- ``optical_flow``: final idle = optical-flow detector.
- ``ml``: final idle = ML anomaly (falls back to optical flow until the rolling
  window is full or when a zone has no trained model).
- ``combined``: fuse both signals via the configured combine strategy (also
  falling back to optical flow until the ML window is ready).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.detection.combined import combine_idle
from src.detection.idle_detector import IdleDetector, ZoneDetectionState
from src.ml.inference import MlIdleClassifier, MlZoneResult
from src.pipeline.motion_pipeline import FrameMotion
from src.preprocessing.roi import Zone
from src.utils.config import VALID_MODES


@dataclass(frozen=True)
class ZoneEvaluation:
    """Combined per-zone result for one frame."""

    zone: Zone
    motion_score: float
    mode: str
    optical_flow_state: ZoneDetectionState
    ml_result: MlZoneResult | None
    is_idle: bool
    final_state: ZoneDetectionState


class ModeEvaluator:
    """Evaluate idle state per frame according to the configured mode."""

    def __init__(
        self,
        mode: str,
        detector: IdleDetector,
        classifier: MlIdleClassifier | None = None,
        combine_strategy: str = "and",
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'; expected {list(VALID_MODES)}.")
        if mode in ("ml", "combined") and classifier is None:
            raise ValueError(f"Mode '{mode}' requires an MlIdleClassifier.")
        self._mode = mode
        self._detector = detector
        self._classifier = classifier
        self._combine_strategy = combine_strategy

    @property
    def mode(self) -> str:
        """Active detection mode."""
        return self._mode

    def evaluate_frame(self, frame_motion: FrameMotion) -> list[ZoneEvaluation]:
        """Evaluate every zone for one frame."""
        evaluations: list[ZoneEvaluation] = []
        for zone_motion in frame_motion.zone_motions:
            zone = zone_motion.zone
            of_state = self._detector.update(
                zone_name=zone.name,
                motion_score=zone_motion.motion_score,
                timestamp_seconds=frame_motion.timestamp_seconds,
            )

            ml_result: MlZoneResult | None = None
            if self._classifier is not None and self._mode in ("ml", "combined"):
                ml_result = self._classifier.update(
                    zone.name, zone_motion.motion_score
                )

            final_idle = self._decide(of_state, ml_result)
            final_state = (
                of_state
                if final_idle == of_state.is_idle
                else replace(of_state, is_idle=final_idle)
            )
            evaluations.append(
                ZoneEvaluation(
                    zone=zone,
                    motion_score=zone_motion.motion_score,
                    mode=self._mode,
                    optical_flow_state=of_state,
                    ml_result=ml_result,
                    is_idle=final_idle,
                    final_state=final_state,
                )
            )
        return evaluations

    def _decide(
        self,
        of_state: ZoneDetectionState,
        ml_result: MlZoneResult | None,
    ) -> bool:
        if self._mode == "optical_flow":
            return of_state.is_idle
        # ml / combined: fall back to optical flow until the ML window is ready.
        if ml_result is None or not ml_result.window_ready:
            return of_state.is_idle
        if self._mode == "ml":
            return ml_result.is_anomaly
        return combine_idle(
            of_state.is_idle, ml_result.is_anomaly, self._combine_strategy
        )
