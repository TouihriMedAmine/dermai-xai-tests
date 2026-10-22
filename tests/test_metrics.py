import numpy as np
import pytest

from dermai_xai.metrics import classification_metrics, expected_calibration_error


def test_ece_zero_for_perfect_confidence():
    probas = np.eye(3)
    labels = np.array([0, 1, 2])
    assert expected_calibration_error(probas, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_one_for_all_wrong_and_confident():
    probas = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    labels = np.array([0, 1, 1])
    assert expected_calibration_error(probas, labels) == pytest.approx(1.0, abs=1e-9)


def test_ece_matches_weighted_interval_error():
    probas = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    labels = np.array([0, 1, 2])
    expected = 1.0 - 1.0 / 3.0
    assert expected_calibration_error(probas, labels) == pytest.approx(expected, abs=1e-9)


def test_ece_zero_for_chance_confidence():
    probas = np.full((10, 2), 0.5)
    labels = np.array([0, 1] * 5)
    assert expected_calibration_error(probas, labels) == pytest.approx(0.0, abs=1e-9)


def test_classification_metrics_perfect_predictions():
    probas = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    labels = np.array([0, 1, 2, 0, 1, 2])
    predicted = labels
    report = classification_metrics(labels, predicted, probas)
    assert report["balanced_accuracy"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx(1.0)
    assert report["roc_auc"] == pytest.approx(1.0)
    assert report["ece"] == pytest.approx(0.0, abs=1e-9)


def test_classification_metrics_imperfect_predictions():
    rng = np.random.RandomState(0)
    probas = rng.dirichlet([1.0, 1.0, 1.0], size=30)
    predicted = probas.argmax(axis=1)
    labels = rng.randint(0, 3, size=30)
    report = classification_metrics(labels, predicted, probas)
    assert set(report) == {"balanced_accuracy", "macro_f1", "roc_auc", "pr_auc", "ece"}
    assert 0.0 <= report["balanced_accuracy"] <= 1.0
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert report["roc_auc"] is None or 0.0 <= report["roc_auc"] <= 1.0
    assert report["pr_auc"] is None or 0.0 <= report["pr_auc"] <= 1.0
    assert 0.0 <= report["ece"] <= 1.0
