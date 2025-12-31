import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict

import pandas as pd

# opcional: networkx para centralidad
import networkx as nx


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_dt_utc(x) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="coerce")


@dataclass
class RCAConfig:
    top_k: int = 3
    w_magnitude: float = 1.0
    w_centrality: float = 0.6
    w_earliness: float = 0.4


def load_incidents(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    required = {"incident_id", "run_id", "service", "ts", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Incidents file missing columns: {sorted(missing)}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def load_graph(path: str) -> nx.DiGraph:
    """
    Espera JSON con edges:
    {"edges":[{"src":"auth","dst":"orders","weight":10}, ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        g = json.load(f)

    G = nx.DiGraph()
    for e in g.get("edges", []):
        src = e["src"]
        dst = e["dst"]
        w = float(e.get("weight", 1.0))
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += w
        else:
            G.add_edge(src, dst, weight=w)
    return G


def compute_centrality(G: nx.DiGraph) -> Dict[str, float]:
    if len(G.nodes) == 0:
        return {}
    pr = nx.pagerank(G, weight="weight")
    # normaliza a 0..1
    mx = max(pr.values()) if pr else 1.0
    return {k: (v / (mx + 1e-12)) for k, v in pr.items()}


def rca_rank(
    incident_df: pd.DataFrame,
    G: nx.DiGraph,
    cfg: RCAConfig,
) -> pd.DataFrame:
    """
    incident_df: filas de un solo incident_id, con columnas:
      service, ts, score (score=severidad de alerta/anomalía/predicción)
    """
    # 1) magnitud por servicio
    mag = incident_df.groupby("service")["score"].mean().to_dict()

    # 2) centralidad
    cent = compute_centrality(G)

    # 3) earliness: servicios que aparecen antes dentro del incidente
    t0 = incident_df["ts"].min()
    first_ts = incident_df.groupby("service")["ts"].min().to_dict()
    early = {}
    for svc, t in first_ts.items():
        dt = (t - t0).total_seconds()
        # cuanto menor dt, mayor score
        early[svc] = 1.0 / (1.0 + dt)

    # normaliza early a 0..1
    if early:
        mx = max(early.values())
        early = {k: v / (mx + 1e-12) for k, v in early.items()}

    rows = []
    for svc in sorted(set(incident_df["service"].tolist())):
        s_mag = float(mag.get(svc, 0.0))
        s_cent = float(cent.get(svc, 0.0))
        s_early = float(early.get(svc, 0.0))

        score = (
            cfg.w_magnitude * s_mag
            + cfg.w_centrality * s_cent
            + cfg.w_earliness * s_early
        )

        rows.append(
            {
                "service": svc,
                "score": float(score),
                "magnitude": s_mag,
                "centrality": s_cent,
                "earliness": s_early,
            }
        )

    out = pd.DataFrame(rows).sort_values("score", ascending=False).head(cfg.top_k)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--incidents", required=True, help="incidents_<run_id>.parquet/csv")
    p.add_argument("--graph", required=True, help="service_graph_<run_id>.json")
    p.add_argument("--incident-id", required=True, help="incident_id a explicar")
    p.add_argument("--out-dir", default="models/rca/out")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    ensure_dir(args.out_dir)

    df = load_incidents(args.incidents)
    df = df[df["incident_id"].astype(str) == str(args.incident_id)].copy()
    if df.empty:
        raise ValueError(f"No existe incident_id={args.incident_id} en incidents.")

    G = load_graph(args.graph)

    cfg = RCAConfig(top_k=args.top_k)
    ranked = rca_rank(df, G, cfg)

    out_json = {
        "incident_id": str(args.incident_id),
        "run_id": str(df["run_id"].iloc[0]),
        "n_alerts": int(len(df)),
        "candidates": ranked.to_dict(orient="records"),
    }

    out_path = os.path.join(args.out_dir, f"rca_{args.incident_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"[ok] saved: {out_path}")
    print(ranked.to_string(index=False))


if __name__ == "__main__":
    main()
