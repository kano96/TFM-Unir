from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import subprocess
import json

APP_PORT = int(os.getenv("PORT", "8002"))

INCIDENTS_PATH = os.getenv(
    "INCIDENTS_PATH", "/app/data/processed/incidents_20251215T013009Z.parquet"
)
GRAPH_PATH = os.getenv(
    "GRAPH_PATH", "/app/data/processed/service_graph_20251215T013009Z.json"
)
OUT_DIR = os.getenv("RCA_OUT_DIR", "/app/models/rca/out")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "3"))

app = FastAPI(title="RCA Service")


class RCARequest(BaseModel):
    incident_id: str
    top_k: int | None = None


@app.on_event("startup")
def _startup():
    os.makedirs(OUT_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "incidents_path": INCIDENTS_PATH,
        "graph_path": GRAPH_PATH,
        "out_dir": OUT_DIR,
        "top_k_default": TOP_K_DEFAULT,
        "rca_script_exists": os.path.exists("/app/models/rca/rca.py"),
    }


@app.post("/rca")
def run_rca(req: RCARequest):
    incident_id = req.incident_id.strip()
    if not incident_id:
        raise HTTPException(status_code=422, detail="incident_id es requerido")

    top_k = req.top_k or TOP_K_DEFAULT

    # Ejecuta el mismo script que ya validaste
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
        raise HTTPException(
            status_code=500,
            detail={
                "error": "RCA script failed",
                "stdout": e.stdout[-2000:],
                "stderr": e.stderr[-2000:],
            },
        )

    out_file = os.path.join(OUT_DIR, f"rca_{incident_id}.json")
    if not os.path.exists(out_file):
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

    return {
        "incident_id": incident_id,
        "top_k": top_k,
        "result": payload,
        "stdout_tail": p.stdout.splitlines()[-20:],
    }
