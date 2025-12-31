import argparse
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_features(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {"run_id", "service", "window_start", "window_end"}
    if not required.issubset(df.columns):
        raise ValueError(f"Features deben contener {required}")

    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True)
    return df


def load_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["start_dt"] = pd.to_datetime(df["start_ts"], utc=True)
    df["end_dt"] = pd.to_datetime(df["end_ts"], utc=True)
    return df


def overlaps(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime
) -> bool:
    return (a_start <= b_end) and (b_start <= a_end)


def build_future_label(
    features: pd.DataFrame, labels: pd.DataFrame, horizon_min: int
) -> pd.DataFrame:
    """
    y_future = 1 si existe un fallo para (run_id,service) que ocurra dentro del
    intervalo FUTURO [window_end, window_end + horizon] Y la ventana actual NO está
    ya dentro de un fallo.

    - is_current_fault = solapa [window_start, window_end] con [fault_start, fault_end]
    - is_future_fault  = solapa [window_end, window_end +
    horizon] con [fault_start, fault_end]
    - y_future = is_future_fault AND NOT is_current_fault
    """
    horizon = timedelta(minutes=horizon_min)

    # Pre-index faults por (run_id, service) para evitar filtrar el DF en cada fila
    faults_map: Dict[Tuple[str, str], List[Tuple[datetime, datetime]]] = {}
    for _, r in labels.iterrows():
        key = (str(r["run_id"]), str(r["service"]))
        fs = r["start_dt"]
        fe = r["end_dt"]
        # asegurar tz-aware UTC
        if getattr(fs, "tzinfo", None) is None:
            fs = fs.replace(tzinfo=timezone.utc)
        if getattr(fe, "tzinfo", None) is None:
            fe = fe.replace(tzinfo=timezone.utc)
        faults_map.setdefault(key, []).append((fs, fe))

    y = []
    for _, r in features.iterrows():
        key = (str(r["run_id"]), str(r["service"]))

        ws = r["window_start"].to_pydatetime()
        we = r["window_end"].to_pydatetime()

        if getattr(ws, "tzinfo", None) is None:
            ws = ws.replace(tzinfo=timezone.utc)
        if getattr(we, "tzinfo", None) is None:
            we = we.replace(tzinfo=timezone.utc)

        # ventana actual
        current_start, current_end = ws, we
        # ventana futura (horizonte)
        future_start, future_end = we, we + horizon

        is_current_fault = 0
        is_future_fault = 0

        for fs, fe in faults_map.get(key, []):
            if overlaps(current_start, current_end, fs, fe):
                is_current_fault = 1
            if overlaps(future_start, future_end, fs, fe):
                is_future_fault = 1
            if is_current_fault and is_future_fault:
                break

        y_future = 1 if (is_future_fault == 1 and is_current_fault == 0) else 0
        y.append(y_future)

    out = features.copy()
    out["y_future"] = y
    return out


def select_feature_columns(df: pd.DataFrame):
    exclude = {"run_id", "service", "window_start", "window_end", "y_future"}
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def temporal_split(df: pd.DataFrame, train_ratio=0.7):
    df = df.sort_values("window_start")
    n = int(len(df) * train_ratio)
    return df.iloc[:n], df.iloc[n:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--horizon-min", type=int, default=5)
    parser.add_argument("--out-dir", default="models/artifacts")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    feats = load_features(args.features)
    labels = load_labels(args.labels)

    feats = build_future_label(feats, labels, args.horizon_min)
    feature_cols = select_feature_columns(feats)

    train_df, test_df = temporal_split(feats)

    X_train = train_df[feature_cols]
    y_train = train_df["y_future"]
    X_test = test_df[feature_cols]
    y_test = test_df["y_future"]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    artifact = {
        "model": model,
        "threshold": 0.05,
        "feature_columns": feature_cols,
    }

    joblib.dump(artifact, os.path.join(args.out_dir, "predictor.joblib"))
    with open(os.path.join(args.out_dir, "predictor_features.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("[ok] Predictor entrenado y guardado")


if __name__ == "__main__":
    main()
