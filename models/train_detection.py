import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ID_COLS = {"run_id", "service", "ts"}


def parse_ts(s):
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if "window_start" in df.columns:
            df["ts"] = pd.to_datetime(df["window_start"], utc=True)
        else:
            raise ValueError("El parquet de features debe tener 'ts' o 'window_start'.")
    else:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_labels(path: str) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"run_id", "service", "fault_type", "start_ts", "end_ts"}
    if not required.issubset(labels.columns):
        raise ValueError(f"labels debe contener columnas: {sorted(required)}")

    labels["start_ts"] = parse_ts(labels["start_ts"])
    labels["end_ts"] = parse_ts(labels["end_ts"])
    return labels


def label_anomalies(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Marca y=1 si (run_id, service, ts) cae dentro de cualquier ventana de fallo.
    """
    features = features.copy()
    features["y"] = 0

    # index por run+service para acelerar
    for (run_id, service), g in labels.groupby(["run_id", "service"]):
        fmask = (features["run_id"] == run_id) & (features["service"] == service)
        if not fmask.any():
            continue

        # para cada ventana de fallo, marca
        for _, row in g.iterrows():
            wmask = (
                fmask
                & (features["ts"] >= row["start_ts"])
                & (features["ts"] <= row["end_ts"])
            )
            features.loc[wmask, "y"] = 1

    return features


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in ID_COLS and c != "y"]
    # deja solo numéricas
    numeric_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    if not numeric_cols:
        raise ValueError("No se detectaron columnas numéricas de features.")
    return numeric_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Parquet único de features")
    ap.add_argument("--labels", required=True, help="CSV fault_windows.csv")
    ap.add_argument(
        "--out-dir", default="models/artifacts", help="Directorio de salida"
    )
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    feat = load_features(args.features)
    labels = load_labels(args.labels)
    feat = label_anomalies(feat, labels)

    feature_cols = get_feature_columns(feat)

    # train SOLO con normal
    train_df = feat[feat["y"] == 0].copy()
    if len(train_df) < 50:
        raise ValueError("Muy pocos ejemplos normales para entrenar. Genera más runs.")

    X_train = (
        train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = IsolationForest(
        contamination=args.contamination,
        random_state=args.random_state,
        n_estimators=200,
    )
    model.fit(X_train_s)

    model_path = os.path.join(args.out_dir, "detector.joblib")
    joblib.dump(
        {"scaler": scaler, "model": model, "feature_cols": feature_cols}, model_path
    )

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "contamination": args.contamination,
        "random_state": args.random_state,
        "n_train_normal": int(len(train_df)),
        "n_total": int(len(feat)),
        "feature_cols": feature_cols,
    }
    meta_path = os.path.join(args.out_dir, "detector_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[ok] modelo guardado en: {model_path}")
    print(f"[ok] metadata guardada en: {meta_path}")
    print(f"[ok] features usadas: {len(feature_cols)}")


if __name__ == "__main__":
    main()
