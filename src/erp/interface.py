"""ERP integration contract for future production schedule context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ProductionOrder:
    """Production order context from an ERP system."""

    zone: str
    order_id: str
    product_code: str
    planned_start: str
    planned_end: str


class ErpClient(ABC):
    """Interface for future ERP integrations."""

    @abstractmethod
    def current_order(self, zone: str) -> ProductionOrder | None:
        """Return current production order for a zone, if available."""


class NullErpClient(ErpClient):
    """No-op ERP client used by the video-only MVP."""

    def current_order(self, zone: str) -> ProductionOrder | None:
        """Return no order because ERP integration is not enabled."""
        return None


class MockErpClient(ErpClient):
    """In-memory ERP client for demos and tests (Method 2 placeholder).

    A zone with no current order is interpreted downstream as a *planned stop*
    (no production scheduled, so idle time is expected and not an anomaly). Real
    deployments would replace this with a client querying the ERP/MES API while
    keeping the :class:`ErpClient` contract unchanged.
    """

    def __init__(self, orders_by_zone: dict[str, ProductionOrder] | None = None) -> None:
        self._orders_by_zone = dict(orders_by_zone or {})

    def set_order(self, zone: str, order: ProductionOrder | None) -> None:
        """Set (or clear, with ``None``) the active order for a zone."""
        if order is None:
            self._orders_by_zone.pop(zone, None)
        else:
            self._orders_by_zone[zone] = order

    def current_order(self, zone: str) -> ProductionOrder | None:
        """Return the active production order for a zone, if any."""
        return self._orders_by_zone.get(zone)

