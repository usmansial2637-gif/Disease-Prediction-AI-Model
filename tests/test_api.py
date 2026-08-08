"""
Smoke tests for api.py.

These only exercise endpoints that work whether or not models have been
trained yet (root + /datasets, which reports "not trained yet" gracefully),
so this suite runs standalone in CI without requiring `python main.py` first.
"""
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_root_endpoint_ok():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body


def test_datasets_endpoint_lists_all_three_datasets():
    response = client.get("/datasets")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"breast_cancer", "heart", "diabetes"}


def test_predict_unknown_dataset_returns_error_not_crash():
    response = client.post(
        "/predict",
        json={"dataset": "not_a_real_dataset", "features": {}},
    )
    # Whatever the exact status code, it must fail cleanly (4xx), not 500.
    assert 400 <= response.status_code < 500
