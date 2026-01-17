from collections import defaultdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import subprocess
import json
import time

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response


_published_labelsets = defaultdict(set)  # incident_id -> set[(service, rank)]

APP_PORT = int(os.getenv("PORT", "8002"))

INCIDENTS_PATH = os.getenv(
    "INCIDENTS_PATH", "/app/data/processed/incidents_20251215T013009Z.parquet"
)
GRAPH_PATH = os.getenv(
    "GRAPH_PATH", "/app/data/processed/service_graph_20251215T013009Z.json"
)
OUT_DIR = os.getenv("RCA_OUT_DIR", "/app/models/rca/out")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "3"))

# Evita alta cardinalidad: limita incident_id que expones
EXPOSE_LAST_N_INCIDENTS = int(os.getenv("RCA_EXPOSE_LAST_N_INCIDENTS", "20"))

app = FastAPI(title="RCA Service")


# -----------------------------
# Prometheus metrics
# -----------------------------
RCA_REQUESTS = Counter(
    "aiops_rca_requests_total",
    "Total requests to RCA endpoints",
    ["endpoint", "method", "status"],
)

RCA_ERRORS = Counter(
    "aiops_rca_errors_total",
    "Total RCA errors",
    ["type"],
)

RCA_LATENCY = Histogram(
    "aiops_rca_request_latency_seconds",
    "Latency of RCA requests in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
)

RCA_LAST_RUN_SUCCESS = Gauge(
    "aiops_rca_last_run_success",
    "1 if last RCA run succeeded, else 0",
)

RCA_LAST_RUN_TIMESTAMP = Gauge(
    "aiops_rca_last_run_timestamp",
    "Unix timestamp of last RCA run attempt",
)

# Top-K candidates
RCA_CANDIDATE_SCORE = Gauge(
    "aiops_rca_candidate_score",
    "RCA candidate total score (top-k only)",
    ["incident_id", "service", "rank"],
)

RCA_CANDIDATE_MAGNITUDE = Gauge(
    "aiops_rca_candidate_magnitude",
    "RCA candidate magnitude component (top-k only)",
    ["incident_id", "service", "rank"],
)

RCA_CANDIDATE_CENTRALITY = Gauge(
    "aiops_rca_candidate_centrality",
    "RCA candidate centrality component (top-k only)",
    ["incident_id", "service", "rank"],
)

RCA_CANDIDATE_EARLINESS = Gauge(
    "aiops_rca_candidate_earliness",
    "RCA candidate earliness component (top-k only)",
    ["incident_id", "service", "rank"],
)

# Guardamos los incident_id “recientes” para limpiar métricas viejas
_recent_incidents: list[str] = []


class RCARequest(BaseModel):
    incident_id: str
    top_k: int | None = None


@app.on_event("startup")
def _startup():
    os.makedirs(OUT_DIR, exist_ok=True)
    RCA_LAST_RUN_SUCCESS.set(0)


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    status = 200
    RCA_REQUESTS.labels(endpoint="/health", method="GET", status=str(status)).inc()
    return {
        "status": "ok",
        "incidents_path": INCIDENTS_PATH,
        "graph_path": GRAPH_PATH,
        "out_dir": OUT_DIR,
        "top_k_default": TOP_K_DEFAULT,
        "rca_script_exists": os.path.exists("/app/models/rca/rca.py"),
    }


def _remove_incident_series(incident_id: str) -> None:
    for svc, rank in _published_labelsets.get(incident_id, set()):
        try:
            RCA_CANDIDATE_SCORE.remove(incident_id, svc, rank)
            RCA_CANDIDATE_MAGNITUDE.remove(incident_id, svc, rank)
            RCA_CANDIDATE_CENTRALITY.remove(incident_id, svc, rank)
            RCA_CANDIDATE_EARLINESS.remove(incident_id, svc, rank)
        except KeyError:
            pass
    _published_labelsets.pop(incident_id, None)


def _track_incident(incident_id: str) -> None:
    global _recent_incidents
    if incident_id in _recent_incidents:
        return

    _recent_incidents.append(incident_id)

    if len(_recent_incidents) > EXPOSE_LAST_N_INCIDENTS:
        to_remove = _recent_incidents[:-EXPOSE_LAST_N_INCIDENTS]
        _recent_incidents = _recent_incidents[-EXPOSE_LAST_N_INCIDENTS:]
        for old_id in to_remove:
            _remove_incident_series(old_id)


def _set_candidates_metrics(incident_id: str, candidates: list[dict]) -> None:
    _track_incident(incident_id)

    for idx, c in enumerate(candidates, start=1):
        svc = str(c.get("service", "unknown"))
        rank = str(idx)

        _published_labelsets[incident_id].add((svc, rank))

        RCA_CANDIDATE_SCORE.labels(incident_id=incident_id, service=svc, rank=rank).set(
            float(c.get("score", 0.0))
        )
        RCA_CANDIDATE_MAGNITUDE.labels(
            incident_id=incident_id, service=svc, rank=rank
        ).set(float(c.get("magnitude", 0.0)))
        RCA_CANDIDATE_CENTRALITY.labels(
            incident_id=incident_id, service=svc, rank=rank
        ).set(float(c.get("centrality", 0.0)))
        RCA_CANDIDATE_EARLINESS.labels(
            incident_id=incident_id, service=svc, rank=rank
        ).set(float(c.get("earliness", 0.0)))


@app.post("/rca")
def run_rca(req: RCARequest):
    start = time.time()
    endpoint = "/rca"
    RCA_LAST_RUN_TIMESTAMP.set(time.time())

    incident_id = req.incident_id.strip()
    if not incident_id:
        RCA_ERRORS.labels(type="invalid_incident_id").inc()
        RCA_REQUESTS.labels(endpoint=endpoint, method="POST", status="422").inc()
        raise HTTPException(status_code=422, detail="incident_id es requerido")

    top_k = req.top_k or TOP_K_DEFAULT

    cmd = [
        "python",
        "/app/models/rca/rca.py",
        "--incidents",
        INCIDENTS_PATH,
        "--graph",
        GRAPH_PATH,
        "--incident-id",
        incident_id,
        "--out-dir",
        OUT_DIR,
        "--top-k",
        str(top_k),
    ]

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        RCA_ERRORS.labels(type="rca_script_failed").inc()
        RCA_LAST_RUN_SUCCESS.set(0)
        RCA_REQUESTS.labels(endpoint=endpoint, method="POST", status="500").inc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "RCA script failed",
                "stdout": e.stdout[-2000:],
                "stderr": e.stderr[-2000:],
            },
        )
    except Exception:
        RCA_ERRORS.labels(type="unexpected").inc()
        RCA_LAST_RUN_SUCCESS.set(0)
        RCA_REQUESTS.labels(endpoint=endpoint, method="POST", status="500").inc()
        raise HTTPException(status_code=500, detail="Unexpected error")
    finally:
        RCA_LATENCY.labels(endpoint=endpoint).observe(time.time() - start)

    out_file = os.path.join(OUT_DIR, f"rca_{incident_id}.json")
    if not os.path.exists(out_file):
        RCA_ERRORS.labels(type="output_missing").inc()
        RCA_LAST_RUN_SUCCESS.set(0)
        RCA_REQUESTS.labels(endpoint=endpoint, method="POST", status="500").inc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "RCA output not found",
                "expected": out_file,
                "stdout": p.stdout[-2000:],
            },
        )

    with open(out_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Export Top-K to Prometheus
    candidates = payload.get("candidates", [])
    _set_candidates_metrics(incident_id, candidates)

    RCA_LAST_RUN_SUCCESS.set(1)
    RCA_REQUESTS.labels(endpoint=endpoint, method="POST", status="200").inc()

    return {
        "incident_id": incident_id,
        "top_k": top_k,
        "result": payload,
        "stdout_tail": p.stdout.splitlines()[-20:],
    }
