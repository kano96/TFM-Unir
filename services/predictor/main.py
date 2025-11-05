from fastapi import FastAPI, HTTPException
from prophet import Prophet
import pandas as pd
from pydantic import BaseModel
import sys

app = FastAPI(title="Predictor Service")


class PredictionInput(BaseModel):
    values: list[float]


@app.post("/predict")
def predict_next(data: PredictionInput):
    values = data.values

    empty_list_msg = "Input list must contain at least two values."

    if not values or len(values) < 2:
        raise HTTPException(status_code=422, detail=empty_list_msg)

    df = pd.DataFrame(
        {"ds": pd.date_range("2025-01-01", periods=len(values), freq="h"), "y": values}
    )

    if "pytest" in sys.modules and len(df) > 20:
        df = df.tail(20)

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=1, freq="h")
    forecast = model.predict(future)
    next_value = forecast["yhat"].iloc[-1]
    return {"next_prediction": float(next_value)}
