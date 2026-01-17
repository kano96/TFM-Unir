import os
import time
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response

# -----------------------------
# Config
# -----------------------------
DEFAULT_MODEL_PATH = os.getenv(
    "PREDICTOR_MODEL_PATH", "/app/models/artifacts/predictor.joblib"
)
DEFAULT_THRESHOLD = float(os.getenv("PREDICTOR_THRESHOLD", "0.45"))

app = FastAPI(title="Predictor Service (Supervised)")

MODEL: Any = None
SCALER: Any = None
FEATURE_COLUMNS: Optional[List[str]] = None
THRESHOLD: float = DEFAULT_THRESHOLD


# -----------------------------
# Prometheus metrics
# -----------------------------
PREDICTOR_REQUESTS = Counter(
    "aiops_predictor_requests_total",
    "Total requests to predictor endpoints",
    ["endpoint", "method", "status"],
)

PREDICTOR_ERRORS = Counter(
    "aiops_predictor_errors_total",
    "Total predictor errors",
    ["type"],
)

PREDICTOR_LATENCY = Histogram(
    "aiops_predictor_request_latency_seconds",
    "Latency of predictor requests in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

PREDICTOR_MODEL_LOADED = Gauge(
    "aiops_predictor_model_loaded",
    "1 if the predictor model artifact is loaded, else 0",
)

PREDICTOR_THRESHOLD = Gauge(
    "aiops_predictor_threshold",
    "Current predictor decision threshold (tau)",
)

# These are the key signals for the dashboard:
INCIDENT_PROBABILITY = Gauge(
    "aiops_incident_probability",
    "Predicted probability of incident within horizon",
    ["service"],
)

INCIDENT_FLAG = Gauge(
    "aiops_incident_flag",
    "1 if predicted incident probability >= threshold, else 0",
    ["service"],
)


# -----------------------------
# Schemas
# -----------------------------
class PredictorInput(BaseModel):
    """
    Entrada: vector de features numéricas ya calculadas para una ventana.
    - features: valores numéricos en el MISMO orden de feature_columns si existiera.
    - feature_names (opcional): si lo envías, el servicio puede reordenar / validar.
    - service (opcional): etiqueta para métricas (Grafana por servicio).
    """

    features: List[float] = Field(..., min_length=1)
    feature_names: Optional[List[str]] = None
    service: str = Field(default="unknown", min_length=1)


class PredictorOutput(BaseModel):
    probability: float
    threshold: float
    will_incident_within_horizon: bool


# -----------------------------
# Load model on startup
# -----------------------------
@app.on_event("startup")
def load_artifact() -> None:
    global MODEL, SCALER, FEATURE_COLUMNS, THRESHOLD

    THRESHOLD = DEFAULT_THRESHOLD
    PREDICTOR_THRESHOLD.set(THRESHOLD)

    if not os.path.exists(DEFAULT_MODEL_PATH):
        MODEL = None
        SCALER = None
        FEATURE_COLUMNS = None
        PREDICTOR_MODEL_LOADED.set(0)
        return

    artifact = joblib.load(DEFAULT_MODEL_PATH)

    if isinstance(artifact, dict):
        MODEL = artifact.get("model")
        SCALER = artifact.get("scaler")
        FEATURE_COLUMNS = artifact.get("feature_columns") or artifact.get("features")
    else:
        MODEL = artifact
        SCALER = None
        FEATURE_COLUMNS = None

    PREDICTOR_MODEL_LOADED.set(1 if MODEL is not None else 0)


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health() -> Dict[str, Any]:
    status = 200
    PREDICTOR_REQUESTS.labels(
        endpoint="/health", method="GET", status=str(status)
    ).inc()

    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_path": DEFAULT_MODEL_PATH,
        "threshold": THRESHOLD,
        "has_scaler": SCALER is not None,
        "n_features_expected": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else None,
    }


def _ensure_model_ready():
    if MODEL is None:
        PREDICTOR_ERRORS.labels(type="model_not_loaded").inc()
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no cargado. "
                f"Verifica PREDICTOR_MODEL_PATH={DEFAULT_MODEL_PATH} "
                "y el volumen/copia en Docker."
            ),
        )


def _vector_from_input(payload: PredictorInput) -> np.ndarray:
    feats = payload.features

    if not feats:
        PREDICTOR_ERRORS.labels(type="empty_features").inc()
        raise HTTPException(status_code=422, detail="features no puede estar vacío.")

    if FEATURE_COLUMNS is not None:
        expected = len(FEATURE_COLUMNS)

        # Caso 1: sin nombres
        if payload.feature_names is None:
            if len(feats) != expected:
                PREDICTOR_ERRORS.labels(type="bad_feature_length").inc()
                raise HTTPException(
                    status_code=422,
                    detail=f"""Se esperaban {expected} features,
                    pero llegaron {len(feats)}.""",
                )
            return np.asarray(feats, dtype=float).reshape(1, -1)

        # Caso 2: con nombres
        if len(payload.feature_names) != len(feats):
            PREDICTOR_ERRORS.labels(type="feature_names_mismatch").inc()
            raise HTTPException(
                status_code=422,
                detail="feature_names y features deben tener la misma longitud.",
            )

        incoming_map = {n: float(v) for n, v in zip(payload.feature_names, feats)}
        ordered = [incoming_map.get(col, 0.0) for col in FEATURE_COLUMNS]
        return np.asarray(ordered, dtype=float).reshape(1, -1)

    return np.asarray(feats, dtype=float).reshape(1, -1)


def _predict_proba(x: np.ndarray) -> float:
    if SCALER is not None:
        x = SCALER.transform(x)

    if hasattr(MODEL, "predict_proba"):
        return float(MODEL.predict_proba(x)[:, 1][0])

    if hasattr(MODEL, "decision_function"):
        s = MODEL.decision_function(x)
        s = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-9)
        return float(s[0])

    return float(MODEL.predict(x)[0])


@app.post("/predict", response_model=PredictorOutput)
def predict(payload: PredictorInput):
    start = time.time()
    endpoint = "/predict"

    try:
        _ensure_model_ready()

        x = _vector_from_input(payload)
        prob = _predict_proba(x)

        will_incident = float(prob >= THRESHOLD)

        # Export key values for dashboards
        svc = payload.service or "unknown"
        INCIDENT_PROBABILITY.labels(service=svc).set(prob)
        INCIDENT_FLAG.labels(service=svc).set(will_incident)

        status = 200
        PREDICTOR_REQUESTS.labels(
            endpoint=endpoint, method="POST", status=str(status)
        ).inc()
        return {
            "probability": prob,
            "threshold": THRESHOLD,
            "will_incident_within_horizon": bool(will_incident),
        }

    except HTTPException as e:
        PREDICTOR_REQUESTS.labels(
            endpoint=endpoint, method="POST", status=str(e.status_code)
        ).inc()
        raise
    except Exception:
        PREDICTOR_ERRORS.labels(type="unexpected").inc()
        PREDICTOR_REQUESTS.labels(endpoint=endpoint, method="POST", status="500").inc()
        raise HTTPException(status_code=500, detail="Unexpected error")
    finally:
        PREDICTOR_LATENCY.labels(endpoint=endpoint).observe(time.time() - start)
