"""Binary classification metrics for idle detection (idle = positive class)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class ClassificationMetrics:
    """Accuracy/precision/recall/F1 and the 2x2 confusion matrix.

    The positive class is ``idle``. Confusion counts: ``tp`` = idle predicted
    idle, ``fp`` = active predicted idle, ``fn`` = idle predicted active,
    ``tn`` = active predicted active.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    support: int

    def confusion_matrix(self) -> dict[str, object]:
        """Return a labelled 2x2 confusion matrix (rows = actual)."""
        return {
            "labels": ["active", "idle"],
            "matrix": [[self.tn, self.fp], [self.fn, self.tp]],
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "support": self.support,
            "confusion_matrix": self.confusion_matrix(),
        }


def compute_metrics(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
) -> ClassificationMetrics:
    """Compute idle-detection metrics from aligned boolean sequences."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")
    total = len(y_true)
    if total == 0:
        return ClassificationMetrics(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0)

    matrix = confusion_matrix(list(y_true), list(y_pred), labels=[False, True])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        support=total,
    )
