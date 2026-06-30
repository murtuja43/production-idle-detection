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

