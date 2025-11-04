from fastapi import FastAPI
import numpy as np
from sklearn.ensemble import IsolationForest

app = FastAPI(title="Detector Service")

model = IsolationForest(contamination=0.05, random_state=42)
trained = False


@app.post("/detect")
def detect(values: list[float]):
    global trained
    if not trained:
        model.fit(np.random.randn(100, len(values)))
        trained = True
    score = model.decision_function([values])[0]
    is_anomaly = score < -0.1
    return {"score": float(score), "is_anomaly": is_anomaly}
