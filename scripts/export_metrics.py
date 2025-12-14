import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import pandas as pd
import requests

DEFAULT_PROM_URL = os.getenv("PROM_URL", "http://localhost:9090")
DEFAULT_LABELS_PATH = os.getenv("LABELS_PATH", "data/labels/fault_windows.csv")
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/raw/metrics")
DEFAULT_STEP_SECONDS = int(os.getenv("STEP_SECONDS", "5"))
DEFAULT_PAD_SECONDS = int(os.getenv("PAD_SECONDS", "30"))  # padding antes/después


# -----------------------------
# Helpers
# -----------------------------
def parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def prom_query_range(
    prom_url: str, query: str, start: datetime, end: datetime, step_seconds: int
) -> dict:
    url = f"{prom_url.rstrip('/')}/api/v1/query_range"
    params = {
        "query": query,
        "start": to_rfc3339(start),
        "end": to_rfc3339(end),
        "step": str(step_seconds),
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")
    return payload


def extract_series(payload: dict) -> List[dict]:
    """Convierte query_range result en filas tipo:
    {metric_labels..., ts, value}
    """
    rows: List[dict] = []
    results = payload["data"]["result"]
    for series in results:
        labels = series.get("metric", {})
        for ts, val in series.get("values", []):
            rows.append(
                {
                    **labels,
                    "ts": datetime.fromtimestamp(float(ts), tz=timezone.utc),
                    "value": float(val) if val not in ("NaN", "Inf", "-Inf") else None,
                }
            )
    return rows


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_last_run_id(labels_df: pd.DataFrame) -> str:
    return labels_df["run_id"].sort_values().iloc[-1]


def compute_time_window(
    labels_df: pd.DataFrame, run_id: str, pad_seconds: int
) -> Tuple[datetime, datetime]:
    df = labels_df[labels_df["run_id"] == run_id].copy()
    if df.empty:
        raise ValueError(f"No se encontró run_id={run_id} en {DEFAULT_LABELS_PATH}")

    df["start_dt"] = df["start_ts"].apply(parse_iso)
    df["end_dt"] = df["end_ts"].apply(parse_iso)

    start = df["start_dt"].min() - timedelta(seconds=pad_seconds)
    end = df["end_dt"].max() + timedelta(seconds=pad_seconds)
    return start, end


# -----------------------------
# PromQL catálogo
# -----------------------------
def build_queries(services: List[str]) -> Dict[str, str]:
    """
    Métricas que vamos a exportar. Asumimos que el simulator expone:
      - app_requests_total{service="user", endpoint="/simulate"}
      - app_errors_total{...}
      - app_request_latency_seconds_bucket{...} (histograma)
    """
    svc_re = "|".join(services)

    return {
        # tasa de requests/seg (sobre el counter)
        "rps": (
            f'sum by (service) (rate(app_requests_total{{service=~"{svc_re}",'
            f' endpoint="/simulate"}}[1m]))'
        ),
        # tasa de errores/seg
        "eps": (
            f'sum by (service) (rate(app_errors_total{{service=~"{svc_re}",'
            f' endpoint="/simulate"}}[1m]))'
        ),
        # error ratio = eps / rps (cuidado con división por 0)
        "error_ratio": (
            f"("
            f"sum by (service)"
            f'(rate(app_errors_total{{service=~"{svc_re}", endpoint="/simulate"}}[1m]))'
            f") / clamp_min(("
            f"sum by (service) "
            f'(rate(app_requests_total{{service=~"{svc_re}", endpoint="/simulate"}}'
            f"[1m]))"
            f"), 0.000001)"
        ),
        # p95 latencia usando buckets del histograma
        "p95_latency": (
            f"histogram_quantile(0.95, "
            f"sum by (le, service) "
            f'(rate(app_request_latency_seconds_bucket{{service=~"{svc_re}",'
            f' endpoint="/simulate"}}[1m]))'
            f")"
        ),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export metrics from Prometheus to parquet/csv."
    )
    parser.add_argument(
        "--prom-url", default=DEFAULT_PROM_URL, help="Prometheus base URL"
    )
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH, help="Labels CSV path")
    parser.add_argument(
        "--run-id", default=None, help="Run ID to export (default: last)"
    )
    parser.add_argument(
        "--services",
        default="user,auth,orders",
        help="Comma-separated logical services (must match 'service' label)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP_SECONDS,
        help="query_range step seconds",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=DEFAULT_PAD_SECONDS,
        help="padding seconds around label windows",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    args = parser.parse_args()

    services = [s.strip() for s in args.services.split(",") if s.strip()]
    if not services:
        raise ValueError("No services provided. Use --services user,auth,orders")

    if not os.path.exists(args.labels):
        raise FileNotFoundError(
            f"No existe {args.labels}. Ejecuta primero scripts/inject_faults.py"
        )

    labels_df = pd.read_csv(args.labels)
    required_cols = {"run_id", "service", "fault_type", "start_ts", "end_ts"}
    if not required_cols.issubset(set(labels_df.columns)):
        raise ValueError(f"Labels CSV debe contener columnas: {sorted(required_cols)}")

    run_id = args.run_id or get_last_run_id(labels_df)
    start, end = compute_time_window(labels_df, run_id, args.pad)

    print(f"[export] run_id={run_id}")
    print(
        f"""[export] window
        start={start.isoformat()} end={end.isoformat()} step={args.step}s"""
    )
    print(f"[export] services={services}")
    print(f"[export] prom_url={args.prom_url}")

    queries = build_queries(services)

    all_frames: List[pd.DataFrame] = []
    for metric_name, promql in queries.items():
        print(f"[query] {metric_name} -> {promql}")
        payload = prom_query_range(args.prom_url, promql, start, end, args.step)
        rows = extract_series(payload)
        if not rows:
            print(f"[warn] empty result for metric={metric_name}")
            continue

        df = pd.DataFrame(rows)
        # Si la serie viene sin label "service"
        if "service" not in df.columns:
            df["service"] = "unknown"

        df["metric"] = metric_name
        df["run_id"] = run_id
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError(
            "No se exportaron métricas. Revisa Prometheus /targets y labels."
        )

    out = pd.concat(all_frames, ignore_index=True)

    # Normaliza columnas base
    keep_cols = ["run_id", "ts", "service", "metric", "value"]
    extra_cols = [c for c in out.columns if c not in keep_cols]
    out = out[keep_cols + extra_cols]

    ensure_dir(args.out_dir)
    parquet_path = os.path.join(args.out_dir, f"metrics_{run_id}.parquet")
    csv_path = os.path.join(args.out_dir, f"metrics_{run_id}.csv")

    # Prefer parquet, fallback to CSV
    try:
        out.to_parquet(parquet_path, index=False)
        print(f"[ok] saved parquet: {parquet_path}")
    except Exception as e:
        print(f"[warn] parquet failed ({e}); saving CSV instead")
        out.to_csv(csv_path, index=False)
        print(f"[ok] saved csv: {csv_path}")

    summary = (
        out.groupby(["service", "metric"])["value"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )
    summary_path = os.path.join(args.out_dir, f"metrics_{run_id}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[ok] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
