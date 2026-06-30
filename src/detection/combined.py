"""Fusion of optical-flow and ML idle signals for combined mode."""

from __future__ import annotations

from src.utils.config import VALID_COMBINE_STRATEGIES


def combine_idle(of_idle: bool, ml_anomaly: bool, strategy: str) -> bool:
    """Combine the optical-flow and ML idle signals.

    Args:
        of_idle: Whether the threshold+duration optical-flow detector reports idle.
        ml_anomaly: Whether the Isolation Forest flags the current motion window
            as anomalous (i.e. an unusual / likely-idle window).
        strategy: ``"and"`` (idle only when both agree; high precision) or
            ``"or"`` (idle when either fires; high recall).
    """
    if strategy == "and":
        return of_idle and ml_anomaly
    if strategy == "or":
        return of_idle or ml_anomaly
    raise ValueError(
        f"Unknown combine strategy '{strategy}'; "
        f"expected one of {list(VALID_COMBINE_STRATEGIES)}."
    )
