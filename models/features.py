# models/features.py
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# Config / Helpers
# -----------------------------
UTC = timezone.utc

DEFAULT_METRICS_DIR = os.getenv("METRICS_DIR", "data/raw/metrics")
DEFAULT_LOGS_DIR = os.getenv("LOGS_DIR", "data/raw/logs")
DEFAULT_OUT_DIR = os.getenv("FEATURES_OUT_DIR", "data/processed")
DEFAULT_TZ = os.getenv("TZ", "UTC")

WINDOWS = ["1min", "5min", "15min"]

ERROR_KEYWORDS = [
    "error",
    "exception",
    "timeout",
    "failed",
    "connection",
    "unavailable",
    "refused",
    "traceback",
    "critical",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_dt(ts: str) -> datetime:
    """
    Parse ISO datetime into UTC.
    """
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def safe_read_parquet_or_csv(path_no_ext: str) -> pd.DataFrame:
    """
    Tries to read parquet first, fallback to CSV.
    """
    parquet = f"{path_no_ext}.parquet"
    csv = f"{path_no_ext}.csv"
    if os.path.exists(parquet):
        return pd.read_parquet(parquet)
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No existe {parquet} ni {csv}")


# -----------------------------
# METRICS FEATURES
# -----------------------------
def build_metrics_features(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input expected columns (from export_metrics.py):
      run_id, ts, service, metric, value
    Output:
      features aggregated per service & window_start/window_end for each window size
    """
    df = metrics_df.copy()

    # normalize
    if "ts" not in df.columns:
        raise ValueError("metrics_df debe contener columna 'ts'")
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        # if ts came as string from csv
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    df = df.dropna(subset=["ts", "service", "metric"])
    df["service"] = df["service"].astype(str)

    # pivot -> columns per metric
    pivot = df.pivot_table(
        index=["run_id", "ts", "service"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()

    # Ensure predictable columns
    for m in ["rps", "eps", "error_ratio", "p95_latency"]:
        if m not in pivot.columns:
            pivot[m] = pd.NA

    pivot = pivot.sort_values(["service", "ts"])

    # rolling window per service
    features_all = []
    pivot = pivot.set_index("ts")

    for window in WINDOWS:
        grp = pivot.groupby(["run_id", "service"], group_keys=False)

        rolled = grp[["rps", "eps", "error_ratio", "p95_latency"]].rolling(window)

        agg = rolled.agg(["mean", "std", "min", "max"]).reset_index()

        # flatten columns
        agg.columns = [f"{c0}_{c1}_{window}" if c1 else c0 for (c0, c1) in agg.columns]
        # after reset_index we have: ts + run_id + service
        # rename ts to window_end and create window_start for reference
        agg = agg.rename(columns={"ts": "window_end"})
        agg["window_start"] = agg["window_end"] - pd.to_timedelta(window)

        features_all.append(agg)

    out = pd.concat(features_all, ignore_index=True)
    # put columns in a nicer order
    base = ["run_id", "service", "window_start", "window_end"]
    rest = [c for c in out.columns if c not in base]
    out = out[base + rest]
    return out


# -----------------------------
# LOGS FEATURES
# -----------------------------
def load_logs_jsonl(path: str, run_id: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(
            columns=["run_id", "ts", "service", "level", "message", "line"]
        )

    df = pd.DataFrame(rows)

    # Asegura run_id aunque el JSONL no lo tenga
    if "run_id" not in df.columns:
        df["run_id"] = run_id
    else:
        df["run_id"] = df["run_id"].fillna(run_id)

    # Normaliza timestamps
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    return df


def normalize_log_service(s: str) -> str:
    """
    Export_logs may return 'simulator-user-1' or 'user'.
    Normalize to logical service: user/auth/orders if possible.
    """
    if not isinstance(s, str):
        return "unknown"
    s = s.strip()
    # common patterns
    s = s.replace("simulator-", "")
    s = re.sub(r"-\d+$", "", s)  # remove -1, -2 ...
    return s


def count_error_keywords(text: str) -> int:
    if not isinstance(text, str):
        return 0
    t = text.lower()
    return sum(1 for k in ERROR_KEYWORDS if k in t)


def build_logs_features(
    logs_df: pd.DataFrame,
    tfidf_max_features: int = 50,
) -> pd.DataFrame:
    """
    Build per-window features:
      - total logs
      - count ERROR/WARN
      - error keyword counts
      - TF-IDF features on "message" (or "line") per window
    """
    df = logs_df.copy()
    if df.empty:
        # return empty but with expected columns
        return pd.DataFrame()

    if "ts" not in df.columns:
        raise ValueError("logs_df debe contener columna 'ts'")
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    df = df.dropna(subset=["ts"])
    df["service"] = df.get("service", "unknown").apply(normalize_log_service)

    # Choose text field
    if "message" not in df.columns:
        df["message"] = None
    if "line" not in df.columns:
        df["line"] = None

    df["text"] = df["message"].fillna(df["line"]).fillna("").astype(str)
    df["level"] = df.get("level", "unknown").fillna("unknown").astype(str).str.upper()
    df["err_kw"] = df["text"].apply(count_error_keywords)

    # Create windowed aggregations
    features_all = []

    df = df.sort_values(["service", "ts"]).set_index("ts")

    # columnas booleanas para evitar groupby.apply
    df["is_error"] = (df["level"] == "ERROR").astype(int)
    df["is_warn"] = (df["level"] == "WARN").astype(int)

    for window in WINDOWS:
        g = df.groupby(["run_id", "service"], group_keys=False)

        cnt = g["text"].rolling(window).count().reset_index(name=f"logs_count_{window}")
        err = (
            g["is_error"]
            .rolling(window)
            .sum()
            .reset_index(name=f"logs_error_count_{window}")
        )
        warn = (
            g["is_warn"]
            .rolling(window)
            .sum()
            .reset_index(name=f"logs_warn_count_{window}")
        )
        kw = (
            g["err_kw"]
            .rolling(window)
            .sum()
            .reset_index(name=f"logs_error_keywords_{window}")
        )

        base = (
            cnt.merge(err, on=["run_id", "service", "ts"])
            .merge(warn, on=["run_id", "service", "ts"])
            .merge(kw, on=["run_id", "service", "ts"])
        )

        base = base.rename(columns={"ts": "window_end"})
        base["window_start"] = base["window_end"] - pd.to_timedelta(window)

        features_all.append(base)

    base_features = pd.concat(features_all, ignore_index=True)

    # TF-IDF per window (only for 1min to keep it cheap)
    # Strategy: bucket logs into 1min windows, concatenate text, fit-transform TF-IDF
    one = base_features[["run_id", "service", "window_start", "window_end"]].copy()
    one = one.drop_duplicates()

    # build documents per (run_id, service, window_end)
    df_reset = df.reset_index()
    df_reset["window_end"] = df_reset["ts"].dt.floor("1min")
    docs = (
        df_reset.groupby(["run_id", "service", "window_end"])["text"]
        .apply(lambda x: " ".join(x.tolist()))
        .reset_index()
    )
    docs["window_start"] = docs["window_end"] - pd.to_timedelta("1min")

    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        lowercase=True,
        stop_words=None,  # could set english/spanish stopwords later
    )
    tfidf = vectorizer.fit_transform(docs["text"])

    tfidf_df = pd.DataFrame(
        tfidf.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf.shape[1])],
    )
    docs_out = pd.concat(
        [docs[["run_id", "service", "window_start", "window_end"]], tfidf_df], axis=1
    )

    # merge TF-IDF into base_features only for 1min endpoints
    out = base_features.merge(
        docs_out,
        on=["run_id", "service", "window_start", "window_end"],
        how="left",
    )

    # fill missing tfidf values with 0
    tfidf_cols = [c for c in out.columns if c.startswith("tfidf_")]
    out[tfidf_cols] = out[tfidf_cols].fillna(0.0)

    return out


# -----------------------------
# TRACES FEATURES (placeholder)
# -----------------------------
def build_traces_features(traces_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Placeholder: when you have an export of traces, produce:
      - spans_count per window
      - avg_span_duration per window
      - dependency graph features (in_degree/out_degree)
    For now returns empty dataframe.
    """
    return pd.DataFrame()


# -----------------------------
# MERGE
# -----------------------------
def merge_features(
    metrics_feat: pd.DataFrame,
    logs_feat: pd.DataFrame,
    traces_feat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge on: run_id, service, window_start, window_end
    """
    base_cols = ["run_id", "service", "window_start", "window_end"]

    out = metrics_feat.copy()

    if not logs_feat.empty:
        out = out.merge(logs_feat, on=base_cols, how="left", suffixes=("", "_logs"))

    if not traces_feat.empty:
        out = out.merge(traces_feat, on=base_cols, how="left", suffixes=("", "_traces"))

    # Fill numeric NaNs with 0 (safe for counts & tfidf)
    num_cols = out.select_dtypes(include=["number"]).columns
    out[num_cols] = out[num_cols].fillna(0.0)

    return out


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Build windowed features for AIOps.")
    parser.add_argument("--run-id", required=True, help="run_id, e.g. 20251214T171732Z")
    parser.add_argument("--metrics-dir", default=DEFAULT_METRICS_DIR)
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--tfidf-max-features", type=int, default=50)
    args = parser.parse_args()

    # Load metrics
    metrics_path_base = os.path.join(args.metrics_dir, f"metrics_{args.run_id}")
    metrics_df = safe_read_parquet_or_csv(metrics_path_base)
    metrics_feat = build_metrics_features(metrics_df)

    # Load logs
    logs_path = os.path.join(args.logs_dir, f"logs_{args.run_id}.jsonl")
    if os.path.exists(logs_path):
        logs_df = load_logs_jsonl(logs_path, run_id=args.run_id)
        logs_feat = build_logs_features(
            logs_df, tfidf_max_features=args.tfidf_max_features
        )
    else:
        logs_feat = pd.DataFrame()

    # Traces placeholder
    traces_feat = build_traces_features(None)

    features = merge_features(metrics_feat, logs_feat, traces_feat)

    ensure_dir(args.out_dir)
    out_path = os.path.join(args.out_dir, f"features_{args.run_id}.parquet")
    features.to_parquet(out_path, index=False)
    print(f"[ok] features saved: {out_path}")
    print(f"[ok] rows={len(features)} cols={len(features.columns)}")


if __name__ == "__main__":
    main()
