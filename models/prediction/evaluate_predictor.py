import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)


def parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ensure_dt_utc(x) -> datetime:
    dt = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"No se pudo parsear datetime: {x}")
    return dt.to_pydatetime().astimezone(timezone.utc)


def overlaps(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime
) -> bool:
    return (a_start <= b_end) and (b_start <= a_end)


def load_features(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {"run_id", "service", "window_start", "window_end"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(
            f"Features debe contener {sorted(required)}. Faltan: {missing}"
        )

    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True, errors="coerce")
    if df["window_start"].isna().any() or df["window_end"].isna().any():
        raise ValueError("No se pudieron parsear window_start/window_end.")

    return df


def load_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"run_id", "service", "fault_type", "start_ts", "end_ts"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Labels debe contener {sorted(required)}. Faltan: {missing}")

    df["start_dt"] = df["start_ts"].apply(parse_iso)
    df["end_dt"] = df["end_ts"].apply(parse_iso)
    return df


def build_y_future(
    features: pd.DataFrame, labels: pd.DataFrame, horizon_min: int
) -> pd.DataFrame:
    """
    y_future = 1 si existe un fallo para (run_id,service) que ocurra dentro del
    intervalo FUTURO [window_end, window_end + horizon] Y la ventana actual NO está
    ya dentro de un fallo.

    - is_current_fault = solapa [window_start, window_end] con [fault_start, fault_end]
    - is_future_fault  = solapa [window_end, window_end + horizon] con
    [fault_start, fault_end]
    - y_future = is_future_fault AND NOT is_current_fault
    """
    horizon = timedelta(minutes=horizon_min)

    # Map faults by (run_id, service)
    faults_map: Dict[Tuple[str, str], List[Tuple[datetime, datetime]]] = {}
    for _, r in labels.iterrows():
        key = (str(r["run_id"]), str(r["service"]))
        faults_map.setdefault(key, []).append((r["start_dt"], r["end_dt"]))

    y = []
    for _, r in features.iterrows():
        key = (str(r["run_id"]), str(r["service"]))

        w_start = ensure_dt_utc(r["window_start"])
        w_end = ensure_dt_utc(r["window_end"])

        # Ventana actual (para excluir "ya en incidente")
        current_start = w_start
        current_end = w_end

        # Ventana futura (horizonte de predicción)
        future_start = w_end
        future_end = w_end + horizon

        is_current_fault = 0
        is_future_fault = 0

        for fs, fe in faults_map.get(key, []):
            if overlaps(current_start, current_end, fs, fe):
                is_current_fault = 1
            if overlaps(future_start, future_end, fs, fe):
                is_future_fault = 1

            # micro-optimización: si ya sabemos ambas, salimos
            if is_current_fault and is_future_fault:
                break

        # Predicción real: solo si hay fallo en el futuro
        y_future = 1 if (is_future_fault == 1 and is_current_fault == 0) else 0
        y.append(y_future)

    out = features.copy()
    out["y_future"] = np.array(y, dtype=int)
    return out


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"run_id", "service", "window_start", "window_end", "y_future"}
    cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        raise ValueError("No hay columnas numéricas para usar como X.")
    return cols


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray
) -> dict:
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def tune_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evalúa múltiples thresholds y devuelve métricas por umbral.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    rows = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
        )

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--labels", default="data/labels/fault_windows.csv")
    p.add_argument("--model-path", default="models/artifacts/predictor.joblib")
    p.add_argument("--horizon-min", type=int, default=5)
    p.add_argument("--out-dir", default="models/prediction/evaluation")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_features(args.features)
    labels = load_labels(args.labels)
    df = build_y_future(df, labels, args.horizon_min)

    print("[debug] y_future positive_rate =", float(df["y_future"].mean()))
    print("[debug] y_future counts =", df["y_future"].value_counts().to_dict())

    X_cols = select_feature_columns(df)
    X = df[X_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = df["y_future"].astype(int).to_numpy()

    artifact = joblib.load(args.model_path)
    model = (
        artifact["model"]
        if isinstance(artifact, dict) and "model" in artifact
        else artifact
    )

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        # normaliza scores a rango 0-1 aprox (solo para evaluación)
        s = model.decision_function(X)
        y_score = (s - s.min()) / (s.max() - s.min() + 1e-9)
    else:
        y_score = model.predict(X).astype(float)

    # -----------------------------
    # Threshold tuning
    # -----------------------------
    thr_df = tune_threshold(y, y_score)
    best = thr_df.sort_values("f1", ascending=False).iloc[0]
    best_thr = float(best["threshold"])

    print("\n[threshold tuning] (top 10 by F1)")
    print(thr_df.sort_values("f1", ascending=False).head(10).to_string(index=False))
    print(
        f"""\n[best threshold] τ*={best_thr:.2f}
        precision={best['precision']:.3f}
        recall={best['recall']:.3f}  f1={best['f1']:.3f}"""
    )

    # Guardar tuning
    thr_df.to_csv(os.path.join(args.out_dir, "threshold_tuning.csv"), index=False)

    # Predicción final usando τ*
    y_pred = (y_score >= best_thr).astype(int)

    report = classification_report(y, y_pred, zero_division=0)
    metrics = compute_metrics(y, y_score, y_pred)
    metrics["threshold"] = best_thr

    print(report)
    print("[metrics]", metrics)

    # save
    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.out_dir, "metrics.csv"), index=False
    )
    with open(
        os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report)

    with open(
        os.path.join(args.out_dir, "run_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "features": args.features,
                "labels": args.labels,
                "model_path": args.model_path,
                "horizon_min": args.horizon_min,
                "n_samples": int(len(df)),
                "positive_rate": float(np.mean(y)) if len(y) else 0.0,
                "n_features": int(len(X_cols)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[ok] saved evaluation to: {args.out_dir}")


if __name__ == "__main__":
    main()
