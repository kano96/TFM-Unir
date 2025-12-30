import os
from typing import Dict, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH", "/app/models/artifacts/detector.joblib")

app = FastAPI(title="Detector Service")


class DetectionInput(BaseModel):
    features: Optional[Dict[str, float]] = None
    values: Optional[list[float]] = None


_bundle = None


def load_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.get("/health")
def health():
    ok = os.path.exists(MODEL_PATH)
    return {"status": "ok", "model_loaded": ok, "model_path": MODEL_PATH}


@app.post("/detect")
def detect(data: DetectionInput):
    try:
        bundle = load_bundle()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    scaler = bundle["scaler"]
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    if data.features is not None:
        x = [float(data.features.get(c, 0.0)) for c in feature_cols]
    elif data.values is not None:
        if len(data.values) != len(feature_cols):
            raise HTTPException(
                status_code=422,
                detail=f"""values debe tener {len(feature_cols)}
                elementos (mismo orden del entrenamiento).""",
            )
        x = [float(v) for v in data.values]
    else:
        raise HTTPException(
            status_code=422, detail="Debe enviar 'features' o 'values'."
        )

    X = np.array([x], dtype=float)
    Xs = scaler.transform(X)

    score = float(model.decision_function(Xs)[0])  # mayor = más normal
    anomaly_score = float(-score)

    # Umbral simple: 0
    is_anomaly = bool(score < 0.0)

    return {
        "score": score,
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
        "n_features": len(feature_cols),
    }
