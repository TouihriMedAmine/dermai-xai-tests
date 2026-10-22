from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEFAULT_MODEL_ID = "sgonzalez2000/dermai-efficientnet-b0"
HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_id: str = DEFAULT_MODEL_ID, device: str | None = None) -> nn.Module:
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    return model.to(resolve_device(device))


def load_processor(model_id: str = DEFAULT_MODEL_ID):
    return AutoImageProcessor.from_pretrained(model_id)


@torch.no_grad()
def predict(model: nn.Module, pixel_values: Tensor) -> dict:
    logits = model(pixel_values=pixel_values).logits
    probabilities = torch.softmax(logits, dim=-1)
    classes = probabilities.argmax(dim=-1)
    labels = [model.config.id2label[int(index)] for index in classes]
    return {
        "logits": logits,
        "probabilities": probabilities,
        "classes": classes,
        "labels": labels,
    }


def module_scorer(model: nn.Module) -> Callable[[Tensor], Tensor]:
    def scorer(pixel_values: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.softmax(model(pixel_values=pixel_values).logits, dim=-1)

    return scorer
