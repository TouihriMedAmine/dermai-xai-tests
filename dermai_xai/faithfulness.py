from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

DEFAULT_FRACTIONS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def _pixel_order(heatmap: Tensor) -> Tensor:
    return heatmap.reshape(heatmap.shape[0], -1).argsort(dim=1, descending=True)


def _target_probs(probs: Tensor, target: Tensor) -> Tensor:
    return probs.gather(1, target.unsqueeze(1)).reshape(-1)


def perturbation_curve(
    scorer: Callable[[Tensor], Tensor],
    pixel_values: Tensor,
    heatmap: Tensor,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    start: Tensor | None = None,
    source: Tensor | None = None,
) -> Tensor:
    """Probabilities of the target class at each fraction of perturbed pixels.

    ``start`` is the image at fraction 0 and ``source`` the values pixels are
    progressively overwritten with, in descending saliency order. A deletion
    curve passes the original image and a baseline ``source`` (probabilities
    drop, lower AUC means a more faithful heatmap); an insertion curve passes
    a baseline image as ``start`` and the original as ``source`` (probabilities
    rise, higher AUC is better).
    """
    target = scorer(pixel_values).argmax(dim=-1)
    batch, channels, height, width = pixel_values.shape
    total = height * width
    order = _pixel_order(heatmap)
    current = start.clone() if start is not None else torch.zeros_like(pixel_values)
    source_pixels = source if source is not None else pixel_values
    flat_current = current.reshape(batch, -1).clone()
    flat_source = source_pixels.reshape(batch, -1)
    steps = []
    with torch.no_grad():
        steps.append(_target_probs(scorer(current), target))
        for fraction in fractions[1:]:
            index = order[:, : round(fraction * total)]
            flat_current.scatter_(1, index, flat_source.gather(1, index))
            current = flat_current.view(batch, channels, height, width)
            steps.append(_target_probs(scorer(current), target))
    return torch.stack(steps, dim=0)


def _trapezoid(fractions: tuple[float, ...], curve: Tensor) -> Tensor:
    x = torch.tensor(fractions, dtype=curve.dtype, device=curve.device)
    return torch.trapezoid(curve, x, dim=0) / (x[-1] - x[0])


def deletion_auc(
    scorer: Callable[[Tensor], Tensor],
    pixel_values: Tensor,
    heatmap: Tensor,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    baseline: float = 0.0,
) -> Tensor:
    start = pixel_values.clone()
    source = torch.full_like(pixel_values, baseline)
    curve = perturbation_curve(scorer, pixel_values, heatmap, fractions=fractions, start=start, source=source)
    return _trapezoid(fractions, curve)


def insertion_auc(
    scorer: Callable[[Tensor], Tensor],
    pixel_values: Tensor,
    heatmap: Tensor,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> Tensor:
    start = pixel_values.mean(dim=(2, 3), keepdim=True).expand_as(pixel_values).contiguous()
    curve = perturbation_curve(scorer, pixel_values, heatmap, fractions=fractions, start=start)
    return _trapezoid(fractions, curve)
