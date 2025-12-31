import os
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
# Schemas
# -----------------------------
class PredictorInput(BaseModel):
    """
    Entrada: vector de features numéricas ya calculadas para una ventana.
    - features: valores numéricos en el MISMO orden de feature_columns si existiera.
    - feature_names (opcional): si lo envías, el servicio puede reordenar / validar.
    """

    features: List[float] = Field(..., min_length=1)
    feature_names: Optional[List[str]] = None


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

    if not os.path.exists(DEFAULT_MODEL_PATH):
        # dejamos el servicio "arriba" pero no listo
        MODEL = None
        SCALER = None
        FEATURE_COLUMNS = None
        return

    artifact = joblib.load(DEFAULT_MODEL_PATH)

    # Aceptamos:
    # - artifact == model
    # - artifact == {"model": ..., "scaler": ..., "feature_columns": ...}
    if isinstance(artifact, dict):
        MODEL = artifact.get("model")
        SCALER = artifact.get("scaler")
        FEATURE_COLUMNS = artifact.get("feature_columns") or artifact.get(
            "features"
        )  # por si cambiaste el nombre
    else:
        MODEL = artifact
        SCALER = None
        FEATURE_COLUMNS = None


@app.get("/health")
def health() -> Dict[str, Any]:
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
        raise HTTPException(
            status_code=503,
            detail=f"""Modelo no cargado.
            Verifica PREDICTOR_MODEL_PATH={DEFAULT_MODEL_PATH}
            y el volumen/copia en Docker.""",
        )


def _vector_from_input(payload: PredictorInput) -> np.ndarray:
    """
    Construye X (1, d).
    Si el artefacto incluye FEATURE_COLUMNS y el usuario manda feature_names,
    reordenamos para alinear.
    """
    feats = payload.features

    # Validación básica
    if not feats:
        raise HTTPException(status_code=422, detail="features no puede estar vacío.")

    # Si tenemos feature_columns esperadas, validamos dimensión
    if FEATURE_COLUMNS is not None:
        expected = len(FEATURE_COLUMNS)

        # Caso 1: El cliente NO manda nombres
        if payload.feature_names is None:
            if len(feats) != expected:
                raise HTTPException(
                    status_code=422,
                    detail=f"""Se esperaban {expected}
                    features (según artefacto), pero llegaron {len(feats)}.""",
                )
            x = np.asarray(feats, dtype=float).reshape(1, -1)
            return x

        # Caso 2: El cliente manda nombres -> reordenamos y completamos faltantes con 0
        if len(payload.feature_names) != len(feats):
            raise HTTPException(
                status_code=422,
                detail="feature_names y features deben tener la misma longitud.",
            )

        incoming_map = {
            name: float(val) for name, val in zip(payload.feature_names, feats)
        }
        ordered = [incoming_map.get(col, 0.0) for col in FEATURE_COLUMNS]
        x = np.asarray(ordered, dtype=float).reshape(1, -1)
        return x

    # Si no tenemos columnas esperadas: usamos lo que llegó tal cual
    x = np.asarray(feats, dtype=float).reshape(1, -1)
    return x


def _predict_proba(x: np.ndarray) -> float:
    """
    Devuelve probabilidad de clase positiva (incidente dentro del horizonte).
    """
    # Aplica scaler si existe
    if SCALER is not None:
        x = SCALER.transform(x)

    if hasattr(MODEL, "predict_proba"):
        p = float(MODEL.predict_proba(x)[:, 1][0])
        return p

    # fallback: decision_function -> [0,1] aprox
    if hasattr(MODEL, "decision_function"):
        s = MODEL.decision_function(x)
        s = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-9)
        return float(s[0])

    # último fallback: predict (0/1)
    y = float(MODEL.predict(x)[0])
    return y


@app.post("/predict", response_model=PredictorOutput)
def predict(payload: PredictorInput):
    _ensure_model_ready()

    x = _vector_from_input(payload)
    prob = _predict_proba(x)

    will_incident = bool(prob >= THRESHOLD)
    return {
        "probability": prob,
        "threshold": THRESHOLD,
        "will_incident_within_horizon": will_incident,
    }
