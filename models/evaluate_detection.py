import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


ID_COLS = {"run_id", "service", "ts"}


def parse_ts(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["ts"] = parse_ts(df["ts"])
    if df["ts"].isna().any():
        raise ValueError("Hay ts inválidos en features.")
    return df


def load_labels(path: str) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"run_id", "service", "fault_type", "start_ts", "end_ts"}
    if not required.issubset(labels.columns):
        raise ValueError(f"labels debe contener: {sorted(required)}")
    labels["start_ts"] = parse_ts(labels["start_ts"])
    labels["end_ts"] = parse_ts(labels["end_ts"])
    return labels


def label_anomalies(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["y"] = 0

    for (run_id, service), g in labels.groupby(["run_id", "service"]):
        fmask = (features["run_id"] == run_id) & (features["service"] == service)
        if not fmask.any():
            continue
        for _, row in g.iterrows():
            wmask = (
                fmask
                & (features["ts"] >= row["start_ts"])
                & (features["ts"] <= row["end_ts"])
            )
            features.loc[wmask, "y"] = 1
    return features


def prf(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(p), float(r), float(f)


def baseline_threshold(df: pd.DataFrame, col="error_ratio", thr=0.2):
    if col not in df.columns:
        return None
    y_pred = (df[col].fillna(0.0) >= thr).astype(int).to_numpy()
    return y_pred


def baseline_zscore(df: pd.DataFrame, col="p95_latency", window=12, z=3.0):
    """
    window=12 si tu step es 5s -> 60s aprox. Ajusta según tu step real.
    """
    if col not in df.columns:
        return None

    out = []
    for _, g in df.sort_values("ts").groupby(["run_id", "service"]):
        s = g[col].astype(float).fillna(0.0)
        mean = s.rolling(window=window, min_periods=max(2, window // 2)).mean()
        std = (
            s.rolling(window=window, min_periods=max(2, window // 2))
            .std()
            .replace(0.0, 1e-9)
        )
        zscore = (s - mean) / std
        pred = (zscore.abs() >= z).astype(int)
        out.append(pred)

    return pd.concat(out).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", default="models/artifacts/detector.joblib")
    ap.add_argument("--out-dir", default="models/artifacts/eval")
    ap.add_argument("--threshold-col", default="error_ratio")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--zscore-col", default="p95_latency")
    ap.add_argument("--z-window", type=int, default=12)
    ap.add_argument("--z", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_features(args.features)
    labels = load_labels(args.labels)
    df = label_anomalies(df, labels)

    y_true = df["y"].astype(int).to_numpy()

    # --- Baseline umbral
    thr_pred = baseline_threshold(df, col=args.threshold_col, thr=args.threshold)
    thr_metrics = None
    if thr_pred is not None:
        p, r, f = prf(y_true, thr_pred)
        thr_metrics = {
            "precision": p,
            "recall": r,
            "f1": f,
            "threshold": args.threshold,
        }

    # --- Baseline z-score
    z_pred = baseline_zscore(df, col=args.zscore_col, window=args.z_window, z=args.z)
    z_metrics = None
    if z_pred is not None:
        p, r, f = prf(y_true, z_pred)
        z_metrics = {
            "precision": p,
            "recall": r,
            "f1": f,
            "z": args.z,
            "window": args.z_window,
        }

    # --- Isolation Forest
    bundle = joblib.load(args.model)
    scaler = bundle["scaler"]
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    Xs = scaler.transform(X)

    # decision_function: más alto = más normal. Anomalía si score < 0 típicamente.
    scores = model.decision_function(Xs)
    # convertimos a “anomaly score” para ROC (más alto = más anómalo)
    anomaly_score = -scores

    # Umbral simple sobre score: si anomaly_score > t => anomalía
    # para tener un punto fijo, usa percentil 95 de normal
    normal_scores = anomaly_score[df["y"] == 0]
    if len(normal_scores) == 0:
        raise ValueError("No hay ejemplos normales en el dataset evaluado.")

    t = float(np.percentile(normal_scores, 95))
    if_pred = (anomaly_score >= t).astype(int)

    p, r, f = prf(y_true, if_pred)
    try:
        auc = float(roc_auc_score(y_true, anomaly_score))
    except Exception:
        auc = None

    if_metrics = {"precision": p, "recall": r, "f1": f, "auc": auc, "threshold": t}

    report = {
        "baseline_threshold": thr_metrics,
        "baseline_zscore": z_metrics,
        "isolation_forest": if_metrics,
        "notes": {
            "if_threshold_rule": "threshold = p95(anomaly_score on normal windows)",
        },
    }

    out_json = os.path.join(args.out_dir, "detection_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    out_pred = os.path.join(args.out_dir, "predictions.parquet")
    out_df = df[["run_id", "service", "ts", "y"]].copy()
    out_df["if_anomaly_score"] = anomaly_score
    out_df["if_pred"] = if_pred
    out_df.to_parquet(out_pred, index=False)

    print(f"[ok] reporte: {out_json}")
    print(f"[ok] predicciones: {out_pred}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
