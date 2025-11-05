from fastapi import FastAPI, HTTPException
import numpy as np
from sklearn.ensemble import IsolationForest
from pydantic import BaseModel

app = FastAPI(title="Detector Service")


class DetectionInput(BaseModel):
    values: list[float]


model = IsolationForest(contamination=0.05, random_state=42)
trained = False


@app.post("/detect")
def detect(data: DetectionInput):
    global trained
    values = data.values

    if not values:
        raise HTTPException(
            status_code=422, detail="Debe proporcionar una lista de valores numéricos."
        )

    if not trained:
        model.fit(np.random.randn(100, len(values)))
        trained = True
    score = model.decision_function([values])[0]
    is_anomaly = bool(score < -0.1)
    return {"score": float(score), "is_anomaly": is_anomaly}
