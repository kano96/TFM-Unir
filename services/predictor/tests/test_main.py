import os
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import importlib.util
import pathlib

# load the service's main.py as a module regardless of packaging
main_path = pathlib.Path(__file__).resolve().parents[1] / "main.py"
spec = importlib.util.spec_from_file_location("predictor_main", str(main_path))
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)

# Helpers ----------------


class ConstProbaModel:
    def __init__(self, p: float):
        self._p = float(p)

    def predict_proba(self, x):
        # returns array [[1-p, p]] for each row
        p = np.clip(self._p, 0.0, 1.0)
        arr = np.array([[1 - p, p] for _ in range(x.shape[0])])
        return arr


class SumProbaModel:
    """Produces probability proportional to sum(x)/100 capped to 1.0"""

    def predict_proba(self, x):
        p = np.clip(x.sum(axis=1) / 100.0, 0.0, 1.0)
        return np.vstack([1 - p, p]).T


class DecisionModel:
    def __init__(self, score: float):
        self._score = score

    def decision_function(self, x):
        return np.array([self._score for _ in range(x.shape[0])])


class PredictModel:
    def __init__(self, y: int):
        self._y = int(y)

    def predict(self, x):
        return np.array([self._y for _ in range(x.shape[0])])


class DummyScaler:
    def transform(self, x):
        return x * 2.0


# Tests -------------------------


def teardown_function(fn):
    # reset module-level globals to a clean state between tests
    main.MODEL = None
    main.SCALER = None
    main.FEATURE_COLUMNS = None
    main.THRESHOLD = main.DEFAULT_THRESHOLD
    main.DEFAULT_MODEL_PATH = os.getenv("PREDICTOR_MODEL_PATH", main.DEFAULT_MODEL_PATH)


def test_health_when_model_missing():
    client = TestClient(main.app)
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["model_loaded"] is False
    assert j["has_scaler"] is False
    assert j["n_features_expected"] is None


def test_predict_returns_503_when_model_not_loaded():
    client = TestClient(main.app)
    r = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert r.status_code == 503
    assert "Modelo no cargado" in r.json().get("detail", "")


def test_load_artifact_and_predict_with_scaler(tmp_path):
    # create artifact with model + scaler + feature_columns and save to disk
    artifact_path = tmp_path / "artifact.joblib"
    model = ConstProbaModel(0.8)
    scaler = DummyScaler()
    artifact = {"model": model, "scaler": scaler, "feature_columns": ["a", "b"]}
    joblib.dump(artifact, str(artifact_path))

    # point module to the artifact and load
    old_path = main.DEFAULT_MODEL_PATH
    main.DEFAULT_MODEL_PATH = str(artifact_path)
    main.load_artifact()

    assert main.MODEL is not None
    assert main.SCALER is not None
    assert main.FEATURE_COLUMNS == ["a", "b"]

    client = TestClient(main.app)
    r = client.get("/health").json()
    assert r["model_loaded"] is True
    assert r["has_scaler"] is True
    assert r["n_features_expected"] == 2

    # predict: scaler will double inputs but model returns constant 0.8
    pr = client.post("/predict", json={"features": [1.0, 2.0]})
    assert pr.status_code == 200
    j = pr.json()
    assert pytest.approx(j["probability"], rel=1e-6) == 0.8
    assert j["threshold"] == main.DEFAULT_THRESHOLD
    assert j["will_incident_within_horizon"] is True

    # restore
    main.DEFAULT_MODEL_PATH = old_path


def test_feature_names_reorder_and_fill_missing():
    main.FEATURE_COLUMNS = ["f1", "f2", "f3"]
    main.SCALER = None
    main.MODEL = SumProbaModel()

    client = TestClient(main.app)

    # send feature_names in different order and omit f2 -> should fill with 0
    payload = {"feature_names": ["f3", "f1"], "features": [10.0, 20.0]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    j = r.json()

    # ordered vector should be [f1=20, f2=0, f3=10] -> sum = 30 -> prob = 0.3
    assert pytest.approx(j["probability"], rel=1e-6) == 0.3


def test_decision_function_and_predict_fallbacks():
    client = TestClient(main.app)

    # decision_function fallback -> normalized over single value -> yields 0.0
    main.MODEL = DecisionModel(42.0)
    r = client.post("/predict", json={"features": [1.0]})
    assert r.status_code == 200
    j = r.json()
    assert pytest.approx(j["probability"], rel=1e-6) == 0.0
    assert j["will_incident_within_horizon"] is False

    # predict fallback -> returns 0/1 as probability
    main.MODEL = PredictModel(1)
    r2 = client.post("/predict", json={"features": [1.0]})
    assert r2.status_code == 200
    j2 = r2.json()
    assert pytest.approx(j2["probability"], rel=1e-6) == 1.0
    assert j2["will_incident_within_horizon"] is True


def test_input_validations(tmp_path):
    # create a small artifact so startup loads a model
    artifact_path = tmp_path / "artifact.joblib"
    joblib.dump({"model": ConstProbaModel(0.1)}, str(artifact_path))

    old_path = main.DEFAULT_MODEL_PATH
    main.DEFAULT_MODEL_PATH = str(artifact_path)
    main.load_artifact()

    client = TestClient(main.app)

    # empty features -> 422
    r = client.post("/predict", json={"features": []})
    assert r.status_code == 422

    # when FEATURE_COLUMNS defined and client doesn't send names -> length must match
    main.FEATURE_COLUMNS = ["a", "b", "c"]
    r2 = client.post("/predict", json={"features": [1.0, 2.0]})
    assert r2.status_code == 422
    assert "Se esperaban" in r2.json().get("detail", "")

    # mismatched feature_names/features length -> 422
    r3 = client.post("/predict", json={"feature_names": ["a", "b"], "features": [1.0]})
    assert r3.status_code == 422
    assert "feature_names y features" in r3.json().get("detail", "")

    # restore
    main.DEFAULT_MODEL_PATH = old_path
