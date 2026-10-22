import pytest
import torch

from dermai_xai.faithfulness import deletion_auc, insertion_auc
from dermai_xai.gradcam import GradCAM
from dermai_xai.model import DEFAULT_MODEL_ID, HAM10000_CLASSES, load_model, predict


@pytest.fixture(scope="module")
def model():
    return load_model(DEFAULT_MODEL_ID, "cpu")


@pytest.mark.network
def test_checkpoint_exposes_ham10000_label_map(model):
    mapping = model.config.id2label
    assert [mapping[index] for index in range(7)] == HAM10000_CLASSES


@pytest.mark.network
def test_checkpoint_probability_is_normalized(model):
    torch.manual_seed(0)
    batch = torch.rand(2, 3, 224, 224)
    result = predict(model, batch)
    assert result["probabilities"].shape == (2, 7)
    torch.testing.assert_close(
        result["probabilities"].sum(dim=1), torch.ones(2), atol=1e-5, rtol=1e-5
    )
    assert all(label in HAM10000_CLASSES for label in result["labels"])


@pytest.mark.network
def test_gradcam_on_checkpoint(model):
    torch.manual_seed(0)
    batch = torch.rand(1, 3, 224, 224)
    engine = GradCAM(model)
    try:
        output = engine(batch)
        assert output["cam"].shape == (1, 224, 224)
        assert output["cam"].min() >= 0.0
        assert output["cam"].max() <= 1.0 + 1e-6
    finally:
        engine.close()


@pytest.mark.network
def test_gradcam_beats_random_saliency_on_deletion(model):
    torch.manual_seed(0)
    batch = torch.rand(1, 3, 224, 224)
    engine = GradCAM(model)
    try:
        cam = engine(batch)["cam"].detach()
    finally:
        engine.close()
    scorer = lambda pixel_values: torch.softmax(model(pixel_values=pixel_values).logits, dim=-1)
    cam_score = deletion_auc(scorer, batch, cam)[0].item()
    random_score = deletion_auc(scorer, batch, torch.rand_like(cam))[0].item()
    assert cam_score <= random_score + 0.02
