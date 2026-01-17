import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
import re

import pandas as pd
import requests

DEFAULT_LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100")
DEFAULT_LABELS_PATH = os.getenv("LABELS_PATH", "data/labels/fault_windows.csv")
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/raw/logs")
DEFAULT_PAD_SECONDS = int(os.getenv("PAD_SECONDS", "30"))
DEFAULT_LIMIT = int(os.getenv("LOKI_LIMIT", "5000"))


def parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


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


def loki_query_range(
    loki_url: str, query: str, start: datetime, end: datetime, limit: int
) -> dict:
    url = f"{loki_url.rstrip('/')}/loki/api/v1/query_range"
    params = {
        "query": query,
        "start": str(to_ns(start)),
        "end": str(to_ns(end)),
        "limit": str(limit),
        "direction": "forward",  # chronological
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Loki query failed: {payload}")
    return payload


def extract_logs(payload: dict) -> List[dict]:
    """
    Loki devuelve streams con labels y values:
      data.result[i].stream = {label: value}
      data.result[i].values = [[ts_ns, line], ...]
    """
    rows: List[dict] = []
    results = payload.get("data", {}).get("result", [])

    def infer_service_from_container(container: str) -> str:
        # Ej: /repo-simulator-user-1  -> user
        m = re.search(r"simulator-([a-zA-Z0-9_-]+)", container or "")
        return m.group(1) if m else "unknown"

    for s in results:
        labels = s.get("stream", {})
        values = s.get("values", [])

        container = labels.get("container", "")  # /repo-simulator-user-1
        # 1) prioridad: label explícito (si existe)
        base_service = labels.get("service") or labels.get("app") or labels.get("job")

        for ts_ns, line in values:
            ts = datetime.fromtimestamp(int(ts_ns) / 1_000_000_000, tz=timezone.utc)

            parsed = None
            level = None
            message = None
            service = base_service  # puede venir None

            # 2) Si el log es JSON, úsalo como fuente de verdad
            try:
                parsed = json.loads(line)
                level = parsed.get("level") or parsed.get("lvl")
                message = parsed.get("message") or parsed.get("msg")
                # aquí está el fix clave:
                service = parsed.get("service") or service
            except Exception:
                pass

            # 3) Si sigue sin service, inferir por container
            if not service:
                service = infer_service_from_container(container)

            rows.append(
                {
                    "ts": ts,
                    "line": line,
                    "service": service,
                    "level": level,
                    "message": message,
                    "labels": labels,
                }
            )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Export logs from Loki aligned to run_id labels."
    )
    parser.add_argument(
        "--loki-url",
        default=DEFAULT_LOKI_URL,
        help="Loki base URL (default http://localhost:3100)",
    )
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH, help="Labels CSV path")
    parser.add_argument(
        "--run-id", default=None, help="Run ID to export (default: last)"
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
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max log lines per query_range call",
    )
    parser.add_argument(
        "--services",
        default="user,auth,orders",
        help="Comma-separated services to filter in Loki labels (service=...).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        raise FileNotFoundError(
            f"No existe {args.labels}. Ejecuta primero scripts/inject_faults.py"
        )

    labels_df = pd.read_csv(args.labels)
    required_cols = {"run_id", "service", "fault_type", "start_ts", "end_ts"}
    if not required_cols.issubset(set(labels_df.columns)):
        raise ValueError(
            f"Labels CSV debe contener columnas: {sorted(required_cols)}. "
            f"Columnas actuales: {labels_df.columns.tolist()}"
        )

    run_id = args.run_id or get_last_run_id(labels_df)
    start, end = compute_time_window(labels_df, run_id, args.pad)

    services = [s.strip() for s in args.services.split(",") if s.strip()]
    svc_re = "|".join(services)
    query = f'{{service=~"{svc_re}"}}'

    print(f"[export] run_id={run_id}")
    print(
        f"""[export] window start={start.isoformat()}
        end={end.isoformat()} pad={args.pad}s"""
    )
    print(f"[export] loki_url={args.loki_url}")

    # --- intento 1 (service) ---
    print(f"[query] {query}")
    payload = loki_query_range(args.loki_url, query, start, end, args.limit)
    logs = extract_logs(payload)

    ensure_dir(args.out_dir)
    jsonl_path = os.path.join(args.out_dir, f"logs_{run_id}.jsonl")
    summary_path = os.path.join(args.out_dir, f"logs_{run_id}_summary.csv")

    # Guarda JSONL (una línea por evento)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in logs:
            out = {
                "run_id": run_id,
                "ts": r["ts"].isoformat(),
                "service": r["service"],
                "level": r["level"],
                "message": r["message"],
                "line": r["line"],
                "labels": r["labels"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[ok] saved jsonl: {jsonl_path} (rows={len(logs)})")

    # Resumen
    df = pd.DataFrame(
        {
            "service": [r["service"] or "unknown" for r in logs],
            "level": [r["level"] or "unknown" for r in logs],
        }
    )
    if not df.empty:
        summary = df.groupby(["service", "level"]).size().reset_index(name="count")
        summary.to_csv(summary_path, index=False)
        print(f"[ok] saved summary: {summary_path}")
    else:
        print("[warn] no logs returned from Loki (empty result).")


if __name__ == "__main__":
    main()
