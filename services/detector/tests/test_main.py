import os
import numpy as np
import pytest
from fastapi.testclient import TestClient
import sys
import main as detector_main

# ensure the services/detector module can be imported when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


client = TestClient(detector_main.app)


class DummyScaler:
    def transform(self, X):
        return X


class DummyModel:
    def __init__(self, factor=1.0):
        self.factor = factor

    def decision_function(self, X):
        # return a positive score when sum(X) > 0 to indicate "normal"
        return np.array([self.factor * np.sum(X[0])])


def make_bundle(n_features=4, factor=1.0):
    feature_cols = [f"f{i}" for i in range(n_features)]
    return {
        "scaler": DummyScaler(),
        "model": DummyModel(factor=factor),
        "feature_cols": feature_cols,
    }


@pytest.fixture(autouse=True)
def reset_bundle_env(tmp_path, monkeypatch):
    # ensure each test starts with a clean bundle and predictable MODEL_PATH
    detector_main._bundle = None
    monkeypatch.setattr(detector_main, "MODEL_PATH", str(tmp_path / "detector.joblib"))
    yield
    detector_main._bundle = None


def test_health_reports_model_present(tmp_path, monkeypatch):
    # create a dummy file to simulate existing model path
    model_file = tmp_path / "detector.joblib"
    model_file.write_text("dummy")

    monkeypatch.setattr(detector_main, "MODEL_PATH", str(model_file))

    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["model_path"] == str(model_file)


def test_detect_with_values_correct_length_returns_expected_json():
    bundle = make_bundle(n_features=4, factor=1.0)
    detector_main._bundle = bundle

    response = client.post("/detect", json={"values": [1.0, 2.0, 3.0, 4.0]})
    assert response.status_code == 200
    res = response.json()
    assert "score" in res and isinstance(res["score"], float)
    assert "anomaly_score" in res and isinstance(res["anomaly_score"], float)
    assert "is_anomaly" in res and isinstance(res["is_anomaly"], bool)
    assert res["n_features"] == 4

    # score should be positive (normal) for positive inputs
    assert res["score"] > 0
    assert res["anomaly_score"] == pytest.approx(-res["score"])
    assert res["is_anomaly"] is False


def test_detect_with_values_wrong_length_returns_422():
    bundle = make_bundle(n_features=4)
    detector_main._bundle = bundle

    response = client.post("/detect", json={"values": [1.0, 2.0, 3.0]})
    assert response.status_code == 422
    assert "values debe tener" in response.json()["detail"]


def test_detect_with_features_dict_and_missing_keys_fill_zero():
    bundle = make_bundle(n_features=3)
    detector_main._bundle = bundle

    # only provide one feature; others should default to 0.0
    response = client.post("/detect", json={"features": {"f0": 5.0}})
    assert response.status_code == 200
    res = response.json()
    assert res["n_features"] == 3

    # score is sum of features (5.0) so positive
    assert res["score"] > 0
    assert res["is_anomaly"] is False


def test_detect_when_no_model_returns_503(monkeypatch):
    # simulate missing model file and ensure _bundle is None
    monkeypatch.setattr(detector_main, "MODEL_PATH", "/non/existent/path.joblib")
    detector_main._bundle = None

    response = client.post("/detect", json={"values": [0.1, 0.2, 0.3]})
    assert response.status_code == 503


def test_detect_with_empty_input_returns_422():
    bundle = make_bundle(n_features=2)
    detector_main._bundle = bundle

    response = client.post("/detect", json={})
    assert response.status_code == 422


def test_detect_with_incorrect_types_returns_422():
    bundle = make_bundle(n_features=3)
    detector_main._bundle = bundle

    response = client.post("/detect", json={"values": ["a", "b", "c"]})
    assert response.status_code == 422
