import numpy as np
import pytest

from dermai_xai.localization import iou, pointing_game_hit, threshold_heatmap


def test_iou_identical_masks():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    assert iou(mask, mask) == pytest.approx(1.0)


def test_iou_disjoint_masks():
    a = np.zeros((4, 4), dtype=bool)
    a[0, 0] = True
    b = np.zeros((4, 4), dtype=bool)
    b[3, 3] = True
    assert iou(a, b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    a = np.zeros((4, 4), dtype=bool)
    a[0:2, :] = True
    b = np.zeros((4, 4), dtype=bool)
    b[1:3, :] = True
    assert iou(a, b) == pytest.approx(1.0 / 3.0)


def test_iou_both_empty_is_zero():
    empty = np.zeros((4, 4), dtype=bool)
    assert iou(empty, empty) == pytest.approx(0.0)


def test_threshold_heatmap_keeps_requested_fraction():
    heatmap = np.arange(36, dtype=float).reshape(6, 6)
    kept = threshold_heatmap(heatmap, fraction=0.25)
    assert kept.sum() == 9
    assert kept[-1, -1]
    assert not kept[0, 0]


def test_threshold_heatmap_never_empty():
    heatmap = np.full((4, 4), 0.5)
    kept = threshold_heatmap(heatmap, fraction=0.0)
    assert kept.sum() >= 1


def test_pointing_game_hit_inside():
    saliency = np.zeros((4, 4))
    saliency[1, 1] = 1.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    assert pointing_game_hit(saliency, mask) is True


def test_pointing_game_miss_outside():
    saliency = np.zeros((4, 4))
    saliency[3, 3] = 1.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    assert pointing_game_hit(saliency, mask) is False


def test_pointing_game_empty_mask_raises():
    saliency = np.zeros((4, 4))
    empty = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError):
        pointing_game_hit(saliency, empty)
