from fastapi import FastAPI
from prophet import Prophet
import pandas as pd
import numpy as np

app = FastAPI(title="Predictor Service")


@app.post("/predict")
def predict_next(values: list[float]):
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=len(values), freq="H"),
                       "y": values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=1, freq="H")
    forecast = model.predict(future)
    next_value = forecast["yhat"].iloc[-1]
    return {"next_prediction": float(next_value)}
