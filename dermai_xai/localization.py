from __future__ import annotations

import numpy as np


def iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask).astype(bool)
    true = np.asarray(true_mask).astype(bool)
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, true).sum() / union)


def threshold_heatmap(heatmap: np.ndarray, fraction: float = 0.15) -> np.ndarray:
    flat = np.asarray(heatmap).reshape(-1)
    count = max(1, min(flat.size, int(round(fraction * flat.size))))
    threshold = np.partition(flat, flat.size - count)[flat.size - count]
    return np.asarray(heatmap) >= threshold


def pointing_game_hit(saliency: np.ndarray, true_mask: np.ndarray) -> bool:
    mask = np.asarray(true_mask).astype(bool)
    if mask.sum() == 0:
        raise ValueError("empty ground truth mask")
    point = np.unravel_index(np.argmax(saliency), np.asarray(saliency).shape)
    return bool(mask[point])
