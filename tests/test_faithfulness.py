import pytest
import torch
from torch import Tensor

from dermai_xai.faithfulness import DEFAULT_FRACTIONS, deletion_auc, insertion_auc, perturbation_curve


class MeanScorer:
    """Single-class scorer: probability equals the mean pixel value of the input."""

    def __call__(self, pixel_values: Tensor) -> Tensor:
        return pixel_values.mean(dim=(1, 2, 3), keepdim=True).reshape(-1, 1)


@pytest.fixture
def image() -> Tensor:
    gradient = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 63.0
    return gradient


@pytest.fixture
def matching_heatmap(image: Tensor) -> Tensor:
    return image.clone()


@pytest.fixture
def opposed_heatmap(image: Tensor) -> Tensor:
    return 1.0 - image


def test_curve_shape_matches_fractions(image, matching_heatmap):
    curve = perturbation_curve(MeanScorer(), image, matching_heatmap, start=torch.zeros_like(image))
    assert curve.shape == (len(DEFAULT_FRACTIONS), 1)


def test_deletion_favors_matching_heatmap(image, matching_heatmap, opposed_heatmap):
    scorer = MeanScorer()
    matching = deletion_auc(scorer, image, matching_heatmap)
    opposed = deletion_auc(scorer, image, opposed_heatmap)
    assert (matching < opposed).all()


def test_insertion_favors_matching_heatmap(image, matching_heatmap, opposed_heatmap):
    scorer = MeanScorer()
    matching = insertion_auc(scorer, image, matching_heatmap)
    opposed = insertion_auc(scorer, image, opposed_heatmap)
    assert (matching > opposed).all()


def test_aucs_are_bounded(image, matching_heatmap):
    scorer = MeanScorer()
    deleted = deletion_auc(scorer, image, matching_heatmap)
    inserted = insertion_auc(scorer, image, matching_heatmap)
    assert (deleted >= 0).all() and (deleted <= 1).all()
    assert (inserted >= 0).all() and (inserted <= 1).all()


def test_deletion_curve_monotonically_decays(image, matching_heatmap):
    curve = perturbation_curve(
        MeanScorer(), image, matching_heatmap, start=image.clone(), source=torch.zeros_like(image)
    )
    assert curve[0, 0] == pytest.approx(0.5, abs=1e-6)
    assert (curve[1:] <= curve[:-1]).all()
    assert curve[-1, 0] == pytest.approx(0.0, abs=1e-6)


def test_insertion_curve_monotonically_rises(image, matching_heatmap):
    curve = perturbation_curve(
        MeanScorer(), image, matching_heatmap, start=torch.zeros_like(image)
    )
    assert (curve[1:] >= curve[:-1]).all()
    assert curve[-1, 0] == pytest.approx(0.5, abs=1e-6)


def test_baseline_value_controls_deletion_floor(image, matching_heatmap):
    scorer = MeanScorer()
    deleted_zero = deletion_auc(scorer, image, matching_heatmap, baseline=0.0)
    deleted_half = deletion_auc(scorer, image, matching_heatmap, baseline=0.25)
    assert (deleted_half > deleted_zero).all()
