import pytest
import torch
from types import SimpleNamespace
from torch import Tensor

from dermai_xai.gradcam import GradCAM, find_target_layer
from dermai_xai.model import HAM10000_CLASSES


class TinyEfficientNetEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.top_activation = torch.nn.Conv2d(4, 4, 3, padding=1)


class TinyEfficientNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.block = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.encoder = TinyEfficientNetEncoder()
        self.head = torch.nn.Linear(4 * 64 * 64, 7)

    def forward(self, pixel_values=None, **unused):
        top = self.encoder.top_activation(torch.relu(self.block(torch.relu(self.stem(pixel_values)))))
        return SimpleNamespace(logits=self.head(torch.relu(top).reshape(top.shape[0], -1)))


class TinyModelForImageClassification(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.efficientnet = TinyEfficientNet()
        self.config = SimpleNamespace(id2label={index: name for index, name in enumerate(HAM10000_CLASSES)})

    def forward(self, pixel_values=None, **unused):
        return self.efficientnet(pixel_values=pixel_values)


@pytest.fixture
def model() -> TinyModelForImageClassification:
    torch.manual_seed(0)
    return TinyModelForImageClassification()


@pytest.fixture
def batch() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, 64, 64, generator=generator)


def test_target_layer_prefers_named_top_activation(model):
    assert find_target_layer(model) is model.efficientnet.encoder.top_activation


def test_gradcam_shapes_and_range(model, batch):
    engine = GradCAM(model)
    try:
        output = engine(batch)
        cam = output["cam"]
        assert cam.shape == (2, 64, 64)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0 + 1e-6
        assert torch.isfinite(cam).all()
        assert output["logits"].shape == (2, 7)
        assert output["target_class"].shape == (2,)
        assert output["cam"].dtype == batch.dtype
    finally:
        engine.close()


def test_gradcam_explains_requested_class(model, batch):
    engine = GradCAM(model)
    try:
        requested = torch.tensor([3, 5])
        output = engine(batch, requested)
        assert output["target_class"].tolist() == [3, 5]
    finally:
        engine.close()


def test_gradcam_grads_do_not_leak_between_calls(model, batch):
    engine = GradCAM(model)
    try:
        first = engine(batch)
        second = engine(batch.flip(3))
        assert first["cam"].shape == second["cam"].shape
        assert torch.isfinite(first["cam"]).all() and torch.isfinite(second["cam"]).all()
    finally:
        engine.close()
