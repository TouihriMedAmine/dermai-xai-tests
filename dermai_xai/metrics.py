from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def expected_calibration_error(probas: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    probas = np.asarray(probas, dtype=float)
    labels = np.asarray(labels)
    confidence = probas.max(axis=1)
    correct = (probas.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(confidence, edges[1:-1], right=True)
    total = float(confidence.size)
    score = 0.0
    for bin_index in range(bins):
        inside = indices == bin_index
        if not inside.any():
            continue
        score += (inside.sum() / total) * abs(correct[inside].mean() - confidence[inside].mean())
    return score


def classification_metrics(labels, predicted, probas, bins: int = 10) -> dict:
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    probas = np.asarray(probas, dtype=float)
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probas, labels, bins),
    }
    try:
        result["roc_auc"] = float(
            roc_auc_score(labels, probas, multi_class="ovr", average="macro", labels=list(range(probas.shape[1])))
        )
        result["pr_auc"] = float(average_precision_score(labels, probas, average="macro"))
    except ValueError:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result
