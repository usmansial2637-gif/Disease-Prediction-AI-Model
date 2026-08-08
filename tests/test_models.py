"""
Tests for src/models.py — makes sure every registered model is a valid
sklearn-compatible estimator that can fit/predict on tiny synthetic data.
"""
import numpy as np

from src.models import get_models, get_param_grids


def test_get_models_returns_nonempty_dict():
    models = get_models()
    assert isinstance(models, dict)
    assert len(models) >= 3  # at minimum: Logistic Regression, SVM, Random Forest


def test_every_model_supports_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    for name, model in get_models().items():
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,), f"{name} produced unexpected prediction shape"
        assert set(np.unique(preds)).issubset({0, 1}), f"{name} predicted non-binary labels"


def test_param_grids_reference_known_models():
    models = get_models()
    grids = get_param_grids()
    for name in grids:
        assert name in models, f"param grid defined for unknown model '{name}'"
