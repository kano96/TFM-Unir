import os
from typing import Dict, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH", "/app/models/artifacts/detector.joblib")

app = FastAPI(title="Detector Service")

# -----------------------
# Prometheus metrics
# -----------------------
DETECT_REQUESTS = Counter(
    "aiops_detector_requests_total",
    "Total number of detection requests",
)

DETECT_ERRORS = Counter(
    "aiops_detector_errors_total",
    "Total number of detector errors",
)

ANOMALY_SCORE = Gauge(
    "aiops_anomaly_score",
    "Latest anomaly score produced by detector",
    ["service"],
)

ANOMALY_FLAG = Gauge(
    "aiops_anomaly_flag",
    "Latest anomaly flag (1=anomaly, 0=normal)",
    ["service"],
)

DETECT_LATENCY = Histogram(
    "aiops_detector_latency_seconds",
    "Latency of detection endpoint",
)

# -----------------------
# API models
# -----------------------


class DetectionInput(BaseModel):
    features: Optional[Dict[str, float]] = None
    values: Optional[list[float]] = None
    service: Optional[str] = None


_bundle = None


def load_bundle():
    global _bundle
    # If _bundle is set (e.g., by tests), use it directly and skip file check
    if _bundle is not None:
        # Simulate missing model if _bundle is explicitly set to False/None by tests
        if _bundle is False:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        return _bundle
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.get("/health")
def health():
    ok = os.path.exists(MODEL_PATH)
    return {"status": "ok", "model_loaded": ok, "model_path": MODEL_PATH}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/detect")
def detect(data: DetectionInput):
    DETECT_REQUESTS.inc()
    with DETECT_LATENCY.time():
        try:
            try:
                bundle = load_bundle()
            except FileNotFoundError as e:
                DETECT_ERRORS.inc()
                raise HTTPException(status_code=503, detail=str(e))

            # Defensive: if bundle is None or missing keys, treat as model missing
            if not bundle or not all(
                k in bundle for k in ("scaler", "model", "feature_cols")
            ):
                DETECT_ERRORS.inc()
                raise HTTPException(status_code=503, detail="Modelo no disponible")

            scaler = bundle["scaler"]
            model = bundle["model"]
            feature_cols = bundle["feature_cols"]

            service = data.service or "unknown"

            if data.features is not None:
                try:
                    x = [float(data.features.get(c, 0.0)) for c in feature_cols]
                except Exception:
                    raise HTTPException(
                        status_code=422, detail="features debe ser un dict de floats"
                    )
            elif data.values is not None:
                if len(data.values) != len(feature_cols):
                    raise HTTPException(
                        status_code=422,
                        detail=f"values debe tener {len(feature_cols)} elementos",
                    )
                try:
                    x = [float(v) for v in data.values]
                except Exception:
                    raise HTTPException(
                        status_code=422, detail="values debe ser una lista de floats"
                    )
            else:
                raise HTTPException(
                    status_code=422, detail="Debe enviar 'features' o 'values'."
                )

            X = np.array([x], dtype=float)
            Xs = scaler.transform(X)

            score = float(model.decision_function(Xs)[0])  # mayor = más normal
            anomaly_score = float(-score)
            is_anomaly = bool(score < 0.0)

            # ---- Prometheus export ----
            ANOMALY_SCORE.labels(service=service).set(anomaly_score)
            ANOMALY_FLAG.labels(service=service).set(1.0 if is_anomaly else 0.0)

            return {
                "score": score,
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
                "service": service,
                "n_features": len(feature_cols),
            }

        except HTTPException as e:
            raise e
        except Exception as e:
            DETECT_ERRORS.inc()
            # For testability, return 422 if input error, else 500
            if isinstance(e, (TypeError, ValueError, KeyError)):
                raise HTTPException(status_code=422, detail=str(e))
            raise HTTPException(status_code=500, detail=str(e))
