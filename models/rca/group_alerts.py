import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_alerts(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    required = {"run_id", "service", "ts", "signal", "score"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Alerts debe contener {sorted(required)}. Faltan: {missing}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if df["ts"].isna().any():
        raise ValueError("No se pudo parsear ts en alerts.")
    return df


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    required = {"run_id", "service", "window_end"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(
            f"Features debe contener {sorted(required)}. Faltan: {missing}"
        )

    df["window_end"] = pd.to_datetime(df["window_end"], utc=True, errors="coerce")
    if df["window_end"].isna().any():
        raise ValueError("No se pudo parsear window_end en features.")
    return df


def select_tfidf_cols(df: pd.DataFrame) -> List[str]:
    cols = [
        c
        for c in df.columns
        if c.startswith("tfidf_") and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(
        cols, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x
    )


def attach_embeddings(
    alerts: pd.DataFrame, features: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Une por (run_id, service, ts ~ window_end) usando merge_asof por tiempo.
    """
    tfidf_cols = select_tfidf_cols(features)
    if not tfidf_cols:
        # sin embeddings
        a = alerts.copy()
        a["has_embed"] = False
        return a, []

    feats = features[["window_end"] + tfidf_cols].copy()
    feats = feats.sort_values("window_end")

    a = alerts.copy()
    a = a.sort_values("ts")

    out_parts = []
    for (run_id, svc), g in a.groupby(["run_id", "service"], sort=False):
        f = features[
            (features["run_id"].astype(str) == str(run_id))
            & (features["service"].astype(str) == str(svc))
        ][["window_end"] + tfidf_cols].copy()
        if f.empty:
            gg = g.copy()
            for c in tfidf_cols:
                gg[c] = 0.0
            gg["has_embed"] = False
            out_parts.append(gg)
            continue

        gg = pd.merge_asof(
            g.sort_values("ts"),
            f.sort_values("window_end"),
            left_on="ts",
            right_on="window_end",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=60),
        )
        # si no hizo match, NaNs -> 0
        for c in tfidf_cols:
            if c not in gg.columns:
                gg[c] = 0.0
        gg[tfidf_cols] = gg[tfidf_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gg["has_embed"] = True
        out_parts.append(gg)

    out = pd.concat(out_parts, ignore_index=True)
    return out, tfidf_cols


def build_clustering_matrix(
    df: pd.DataFrame, tfidf_cols: List[str], time_weight: float
) -> np.ndarray:
    # feature 1: tiempo en segundos desde t0
    t0 = df["ts"].min()
    t_sec = (df["ts"] - t0).dt.total_seconds().astype(float).to_numpy().reshape(-1, 1)
    t_sec = t_sec * float(time_weight)

    # feature 2: score (severidad) opcional
    score = df["score"].astype(float).to_numpy().reshape(-1, 1)

    # feature 3: embedding TF-IDF (si existe)
    if tfidf_cols:
        emb = df[tfidf_cols].to_numpy()
        X = np.hstack([t_sec, score, emb])
    else:
        X = np.hstack([t_sec, score])

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alerts", required=True, help="alerts_<run_id>.parquet/csv")
    p.add_argument(
        "--features", required=True, help="features_<run_id>.parquet/csv (para TF-IDF)"
    )
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument(
        "--eps", type=float, default=0.8, help="DBSCAN eps en espacio normalizado"
    )
    p.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples")
    p.add_argument(
        "--time-weight",
        type=float,
        default=0.05,
        help="peso del tiempo (segundos) en el clustering",
    )
    args = p.parse_args()

    ensure_dir(args.out_dir)

    alerts = load_alerts(args.alerts)
    feats = load_features(args.features)

    merged, tfidf_cols = attach_embeddings(alerts, feats)

    # clustering por run_id separado (más estable)
    all_parts = []
    for run_id, g in merged.groupby("run_id", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        X = build_clustering_matrix(g, tfidf_cols, time_weight=args.time_weight)

        # escala
        Xs = StandardScaler().fit_transform(X)

        model = DBSCAN(eps=args.eps, min_samples=args.min_samples)
        labels = model.fit_predict(Xs)

        g["cluster_id"] = labels  # -1 = outlier
        # define incident_id: cluster_id + run
        g["incident_id"] = g["cluster_id"].apply(
            lambda c: f"{run_id}_inc{c}" if c >= 0 else f"{run_id}_inc_outlier"
        )
        all_parts.append(g)

    out = pd.concat(all_parts, ignore_index=True)
    out_path = os.path.join(
        args.out_dir, os.path.basename(args.alerts).replace("alerts_", "incidents_")
    )
    if out_path.endswith(".csv"):
        out.to_csv(out_path, index=False)
    else:
        out.to_parquet(out_path, index=False)

    print(f"[ok] saved incidents: {out_path} (rows={len(out)})")
    print(
        out[["run_id", "ts", "service", "signal", "score", "incident_id", "cluster_id"]]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
