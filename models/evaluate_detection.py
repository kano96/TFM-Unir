import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_iso(ts: str) -> datetime:
    """
    fault_windows.csv guarda timestamps tipo '2025-12-14T16:32:55+00:00'
    """
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_utc(dt: pd.Timestamp) -> datetime:
    if dt.tzinfo is None:
        return dt.to_pydatetime().replace(tzinfo=timezone.utc)
    return dt.to_pydatetime().astimezone(timezone.utc)


def load_features(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe features parquet/csv: {path}")

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {"run_id", "service", "window_start", "window_end"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(
            f"Features debe contener columnas {sorted(required)}. Faltan: {missing}"
        )

    # Parse time columns
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True, errors="coerce")

    if df["window_start"].isna().any() or df["window_end"].isna().any():
        raise ValueError("No se pudieron parsear window_start/window_end a datetime.")

    return df


def load_labels(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe labels CSV: {path}")
    df = pd.read_csv(path)

    required = {"run_id", "service", "fault_type", "start_ts", "end_ts"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(
            f"Labels debe contener columnas {sorted(required)}. Faltan: {missing}"
        )

    df["start_dt"] = df["start_ts"].apply(parse_iso)
    df["end_dt"] = df["end_ts"].apply(parse_iso)
    return df


def overlaps(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime
) -> bool:
    """
    True si dos intervalos se solapan.
    """
    return (a_start <= b_end) and (b_start <= a_end)


def label_windows(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna y_true=1 si la ventana (window_start/window_end) solapa con un fault_window.
    Matching por (run_id, service).
    """
    feats = features.copy()
    feats["y_true"] = 0

    # index faults by (run_id, service) to speed up
    faults_map: Dict[Tuple[str, str], List[Tuple[datetime, datetime, str]]] = {}
    for _, r in labels.iterrows():
        key = (str(r["run_id"]), str(r["service"]))
        faults_map.setdefault(key, []).append(
            (r["start_dt"], r["end_dt"], str(r["fault_type"]))
        )

    # label
    y = []
    ft = []
    for _, r in feats.iterrows():
        key = (str(r["run_id"]), str(r["service"]))
        ws = to_utc(r["window_start"])
        we = to_utc(r["window_end"])
        is_fault = 0
        fault_type = None
        for fs, fe, ftype in faults_map.get(key, []):
            if overlaps(ws, we, fs, fe):
                is_fault = 1
                fault_type = ftype
                break
        y.append(is_fault)
        ft.append(fault_type if fault_type else "normal")

    feats["y_true"] = y
    feats["fault_type"] = ft
    return feats


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Selecciona columnas numéricas que NO sean identificadores/tiempo/labels.
    """
    exclude = {
        "run_id",
        "service",
        "window_start",
        "window_end",
        "y_true",
        "fault_type",
    }
    candidates = [c for c in df.columns if c not in exclude]
    # solo numéricas
    num_cols = []
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    if not num_cols:
        raise ValueError(
            "No se encontraron columnas numéricas de features para evaluar."
        )
    return num_cols


def train_val_test_split_time(
    df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal por window_start (por simplicidad global).
    """
    d = df.sort_values("window_start").reset_index(drop=True)
    n = len(d)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = d.iloc[:n_train].copy()
    val = d.iloc[n_train : n_train + n_val].copy()
    test = d.iloc[n_train + n_val :].copy()
    return train, val, test


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None
) -> Dict[str, float]:
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

    # AUCs si hay score continuo
    if y_score is not None:
        # roc_auc requiere ambos labels presentes
        if len(np.unique(y_true)) > 1:
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            except Exception:
                out["roc_auc"] = float("nan")
            try:
                out["pr_auc"] = float(average_precision_score(y_true, y_score))
            except Exception:
                out["pr_auc"] = float("nan")
        else:
            out["roc_auc"] = float("nan")
            out["pr_auc"] = float("nan")
    return out


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_pr(
    y_true: np.ndarray, y_score: np.ndarray, out_dir: str, prefix: str
) -> None:
    # ROC & PR sin seaborn
    if len(np.unique(y_true)) < 2:
        return

    from sklearn.metrics import roc_curve, precision_recall_curve

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{prefix} ROC")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_roc.png"), dpi=150)
    plt.close(fig)

    # PR
    p, r, _ = precision_recall_curve(y_true, y_score)
    fig = plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix} PR")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_pr.png"), dpi=150)
    plt.close(fig)


# -----------------------------
# Baselines
# -----------------------------
def baseline_threshold(
    df: pd.DataFrame, col: str, thr: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    y_pred = 1 si col >= thr
    score = col
    """
    score = df[col].astype(float).to_numpy()
    y_pred = (score >= thr).astype(int)
    return y_pred, score


def baseline_moving_zscore(
    df: pd.DataFrame, col: str, window: int, z_thr: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    z-score con rolling mean/std. score = z
    """
    s = df[col].astype(float)
    mu = s.rolling(window=window, min_periods=max(2, window // 3)).mean()
    sigma = s.rolling(window=window, min_periods=max(2, window // 3)).std()
    z = (s - mu) / (sigma.replace(0, np.nan))
    z = z.fillna(0.0)
    score = z.to_numpy()
    y_pred = (score >= z_thr).astype(int)
    return y_pred, score


# -----------------------------
# Isolation Forest (unsupervised)
# -----------------------------
@dataclass
class IFConfig:
    contamination: float = 0.05
    random_state: int = 42
    n_estimators: int = 200


def fit_isolation_forest(X_train: np.ndarray, cfg: IFConfig) -> IsolationForest:
    model = IsolationForest(
        contamination=cfg.contamination,
        random_state=cfg.random_state,
        n_estimators=cfg.n_estimators,
    )
    model.fit(X_train)
    return model


def if_predict(
    model: IsolationForest, X: np.ndarray, score_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    sklearn decision_function: mayor = más normal.
    Definimos anomaly_score = -decision_function (mayor = más anómalo).
    """
    decision = model.decision_function(X)  # normality score
    anomaly_score = -decision
    y_pred = (anomaly_score >= score_threshold).astype(int)
    return y_pred, anomaly_score


# -----------------------------
# MTTD
# -----------------------------
def compute_mttd(
    labeled_features: pd.DataFrame,
    labels: pd.DataFrame,
    y_pred_col: str,
) -> pd.DataFrame:
    """
    Por cada fault window (labels), calcula:
      - first_detect_ts: primer window_start con y_pred=1 que solape ese fallo
      - mttd_seconds: first_detect_ts - fault_start
    """
    rows = []
    for _, f in labels.iterrows():
        run_id = str(f["run_id"])
        svc = str(f["service"])
        fault_type = str(f["fault_type"])
        start_dt = f["start_dt"]
        end_dt = f["end_dt"]

        df = labeled_features[
            (labeled_features["run_id"].astype(str) == run_id)
            & (labeled_features["service"].astype(str) == svc)
        ].copy()

        if df.empty:
            rows.append(
                {
                    "run_id": run_id,
                    "service": svc,
                    "fault_type": fault_type,
                    "fault_start": start_dt.isoformat(),
                    "fault_end": end_dt.isoformat(),
                    "first_detect_ts": None,
                    "mttd_seconds": None,
                }
            )
            continue

        # ventana solapada con fallo y detectada
        def _is_overlap(row) -> bool:
            ws = to_utc(row["window_start"])
            we = to_utc(row["window_end"])
            return overlaps(ws, we, start_dt, end_dt)

        df["overlap"] = df.apply(_is_overlap, axis=1)
        hits = df[(df["overlap"]) & (df[y_pred_col] == 1)].sort_values("window_start")

        if hits.empty:
            rows.append(
                {
                    "run_id": run_id,
                    "service": svc,
                    "fault_type": fault_type,
                    "fault_start": start_dt.isoformat(),
                    "fault_end": end_dt.isoformat(),
                    "first_detect_ts": None,
                    "mttd_seconds": None,
                }
            )
            continue

        first_ts = to_utc(hits.iloc[0]["window_start"])
        mttd = (first_ts - start_dt).total_seconds()
        rows.append(
            {
                "run_id": run_id,
                "service": svc,
                "fault_type": fault_type,
                "fault_start": start_dt.isoformat(),
                "fault_end": end_dt.isoformat(),
                "first_detect_ts": first_ts.isoformat(),
                "mttd_seconds": float(mttd),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly detection baselines + Isolation Forest."
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Parquet/CSV con features (window_start/window_end).",
    )
    parser.add_argument("--labels", required=True, help="CSV fault_windows.csv")
    parser.add_argument(
        "--out-dir", default="models/evaluation", help="Directorio para outputs"
    )
    parser.add_argument(
        "--model-path",
        default="models/artifacts/detector.joblib",
        help="Ruta al modelo IF joblib",
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Guardar plots (confusion/roc/pr)"
    )

    # Baseline config
    parser.add_argument(
        "--baseline-col",
        default="error_ratio_mean_1min",
        help="Columna usada para baselines",
    )
    parser.add_argument(
        "--baseline-thr",
        type=float,
        default=0.10,
        help="Threshold baseline col>=thr => anomaly",
    )
    parser.add_argument(
        "--zscore-window",
        type=int,
        default=12,
        help="Ventana rolling para z-score (en #filas)",
    )
    parser.add_argument(
        "--zscore-thr", type=float, default=3.0, help="z>=thr => anomaly"
    )

    # Isolation Forest config
    parser.add_argument("--if-contamination", type=float, default=0.05)
    parser.add_argument("--if-n-estimators", type=int, default=200)
    parser.add_argument(
        "--if-score-thr", type=float, default=0.10, help="anomaly_score>=thr => anomaly"
    )

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    features = load_features(args.features)
    labels = load_labels(args.labels)
    labeled = label_windows(features, labels)

    # Split temporal
    train_df, val_df, test_df = train_val_test_split_time(
        labeled, train_ratio=0.6, val_ratio=0.2
    )

    feature_cols = select_feature_columns(labeled)

    # Normalizamos NaNs -> 0 (prototipo)
    for d in (train_df, val_df, test_df):
        d[feature_cols] = d[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # -----------------------------
    # Baselines (solo si existe columna)
    # -----------------------------
    results = []

    baseline_available = (
        args.baseline_col in test_df.columns
        and pd.api.types.is_numeric_dtype(test_df[args.baseline_col])
    )
    if baseline_available:
        y_true = test_df["y_true"].to_numpy()

        y_pred, y_score = baseline_threshold(
            test_df, args.baseline_col, args.baseline_thr
        )
        m = compute_metrics(y_true, y_pred, y_score=y_score)
        m.update(
            {
                "model": "baseline_threshold",
                "details": f"{args.baseline_col}>={args.baseline_thr}",
            }
        )
        results.append(m)

        if args.save_plots:
            plot_confusion(
                y_true,
                y_pred,
                os.path.join(args.out_dir, "baseline_threshold_cm.png"),
                "Baseline Threshold CM",
            )
            plot_roc_pr(y_true, y_score, args.out_dir, "baseline_threshold")

        # z-score
        y_pred2, y_score2 = baseline_moving_zscore(
            test_df, args.baseline_col, args.zscore_window, args.zscore_thr
        )
        m2 = compute_metrics(y_true, y_pred2, y_score=y_score2)
        m2.update(
            {
                "model": "baseline_zscore",
                "details": f"""{args.baseline_col}
                z>={args.zscore_thr} win={args.zscore_window}""",
            }
        )
        results.append(m2)

        if args.save_plots:
            plot_confusion(
                y_true,
                y_pred2,
                os.path.join(args.out_dir, "baseline_zscore_cm.png"),
                "Baseline Z-Score CM",
            )
            plot_roc_pr(y_true, y_score2, args.out_dir, "baseline_zscore")
    else:
        print(
            f"""[warn] No se ejecutan baselines: columna '{args.baseline_col}'
            no existe o no es numérica."""
        )

    # -----------------------------
    # Isolation Forest
    # Train: SOLO ventanas normales (y_true=0) para semi-supervised
    # -----------------------------
    X_train = train_df[train_df["y_true"] == 0][feature_cols].to_numpy()
    if len(X_train) < 10:
        # fallback: usa todo train
        X_train = train_df[feature_cols].to_numpy()

    cfg = IFConfig(
        contamination=args.if_contamination,
        random_state=42,
        n_estimators=args.if_n_estimators,
    )

    artifact = None
    model = None

    if os.path.exists(args.model_path):
        try:
            artifact = joblib.load(args.model_path)
            print(f"[load] loaded artifact: {args.model_path}")

            if isinstance(artifact, dict):
                if "model" not in artifact:
                    raise ValueError("Artifact no contiene clave 'model'")
                model = artifact["model"]
                print("[load] artifact contains model + metadata")
            else:
                model = artifact
                print("[load] artifact is raw model")

        except Exception as e:
            print(f"[warn] no se pudo cargar {args.model_path}: {e}")
            model = None

    if model is None:
        print(
            f"[train] fitting IsolationForest (n={len(X_train)}, d={len(feature_cols)})"
        )
        model = fit_isolation_forest(X_train, cfg)
        # guarda para reproducibilidad
        joblib.dump(model, os.path.join(args.out_dir, "detector_if_trained.joblib"))
        with open(
            os.path.join(args.out_dir, "feature_columns.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    # --- Evaluación en TEST (métricas clásicas) ---
    X_test = test_df[feature_cols].to_numpy()
    y_true = test_df["y_true"].to_numpy()
    y_pred, y_score = if_predict(model, X_test, score_threshold=args.if_score_thr)

    m = compute_metrics(y_true, y_pred, y_score=y_score)
    m.update(
        {
            "model": "isolation_forest",
            "details": f"""score_thr={args.if_score_thr}
            contamination={args.if_contamination}""",
        }
    )
    results.append(m)

    # --- Predicción en todo el run (para MTTD) ---
    labeled_scored = labeled.copy()
    labeled_scored[feature_cols] = (
        labeled_scored[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    X_all = labeled_scored[feature_cols].to_numpy()
    y_pred_all, y_score_all = if_predict(
        model, X_all, score_threshold=args.if_score_thr
    )

    labeled_scored["y_pred_if"] = y_pred_all.astype(int)
    labeled_scored["if_score"] = y_score_all

    # --- MTTD ---
    mttd_df = compute_mttd(
        labeled_features=labeled_scored, labels=labels, y_pred_col="y_pred_if"
    )

    mttd_path = os.path.join(args.out_dir, "mttd_isolation_forest.csv")
    mttd_df.to_csv(mttd_path, index=False)

    # Stats MTTD
    mttd_valid = mttd_df["mttd_seconds"].dropna()
    mttd_mean = float(mttd_valid.mean()) if not mttd_valid.empty else float("nan")
    mttd_median = float(mttd_valid.median()) if not mttd_valid.empty else float("nan")

    # -----------------------------
    # Save results
    # -----------------------------
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.out_dir, "metrics.csv")
    results_df.to_csv(results_path, index=False)

    summary = {
        "features_path": args.features,
        "labels_path": args.labels,
        "n_windows_total": int(len(labeled)),
        "n_test_windows": int(len(test_df)),
        "positive_rate_test": float(test_df["y_true"].mean()) if len(test_df) else 0.0,
        "mttd_mean_seconds_if": mttd_mean,
        "mttd_median_seconds_if": mttd_median,
    }
    with open(
        os.path.join(args.out_dir, "run_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[ok] saved metrics: {results_path}")
    print(f"[ok] saved mttd: {mttd_path}")
    print(f"[ok] saved summary: {os.path.join(args.out_dir, 'run_summary.json')}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
