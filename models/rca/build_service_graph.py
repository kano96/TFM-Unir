import argparse
import json
import os
from collections import defaultdict
from typing import Dict
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_incidents(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    required = {"run_id", "incident_id", "ts", "service"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(
            f"Incidents debe contener {sorted(required)}. Faltan: {missing}"
        )

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if df["ts"].isna().any():
        raise ValueError("No se pudo parsear ts.")
    return df


def build_graph_from_cooccurrence(inc: pd.DataFrame, window_s: int) -> Dict[str, list]:
    """
    Crea edges si servicios aparecen cercanos en tiempo dentro del mismo incident_id.
    window_s controla “co-ocurrencia” temporal.
    """
    edges = defaultdict(float)

    for (run_id, inc_id), g in inc.groupby(["run_id", "incident_id"], sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        services = g["service"].astype(str).tolist()
        times = g["ts"].tolist()

        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                dt = (times[j] - times[i]).total_seconds()
                if dt > window_s:
                    break
                a = services[i]
                b = services[j]
                if a == b:
                    continue
                # edge bidireccional por baseline
                edges[(a, b)] += 1.0
                edges[(b, a)] += 1.0

    out_edges = [
        {"src": k[0], "dst": k[1], "weight": float(w)} for k, w in edges.items()
    ]
    return {
        "edges": out_edges,
        "type": "cooccurrence_baseline",
        "window_seconds": int(window_s),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--incidents", required=True, help="incidents_<run_id>.parquet/csv")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument(
        "--window-seconds", type=int, default=120, help="co-ocurrencia temporal"
    )
    args = p.parse_args()

    ensure_dir(args.out_dir)

    inc = load_incidents(args.incidents)
    g = build_graph_from_cooccurrence(inc, args.window_seconds)

    # name output by run_id (si hay uno solo)
    run_ids = sorted(set(inc["run_id"].astype(str).tolist()))
    suffix = run_ids[0] if len(run_ids) == 1 else "multi"

    out_path = os.path.join(args.out_dir, f"service_graph_{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(g, f, ensure_ascii=False, indent=2)

    print(f"[ok] saved graph: {out_path} (edges={len(g['edges'])})")


if __name__ == "__main__":
    main()
