from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def find_target_layer(model: nn.Module) -> nn.Module:
    named = dict(model.named_modules())
    candidates = [name for name in named if name.endswith(".top_activation")]
    if candidates:
        return named[candidates[0]]
    convs = [name for name, module in named.items() if isinstance(module, nn.Conv2d)]
    if not convs:
        raise ValueError("no conv layer available for CAM")
    return named[convs[-1]]


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else find_target_layer(model)
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module: nn.Module, inputs, output: Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def __call__(self, pixel_values: Tensor, target_class: Tensor | None = None) -> dict:
        pixel_values = pixel_values.detach().clone().requires_grad_(True)
        logits = self.model(pixel_values=pixel_values).logits
        if target_class is None:
            target_class = logits.argmax(dim=-1)
        self.model.zero_grad(set_to_none=True)
        logits.gather(1, target_class.unsqueeze(1)).sum().backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        raw_cam = F.relu((weights * self._activations).sum(dim=1))
        size = pixel_values.shape[-2:]
        cam = F.interpolate(raw_cam.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1)
        low = cam.amin(dim=(1, 2), keepdim=True)
        high = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - low) / (high - low).clamp_min(1e-12)
        return {
            "cam": cam,
            "target_class": target_class.detach(),
            "logits": logits.detach(),
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
