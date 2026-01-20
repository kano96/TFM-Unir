import os
import time
import json
import requests
import pandas as pd

FAULT_URL = "http://localhost:8102/fault/errors"

BUILD_ALERTS_CMD = (
    "python models/rca/build_alerts.py "
    "--features {features} "
    "--out-dir data/processed "
    "--run-id {run_id} "
    "--use-metric-rule "
    "--use-log-rule"
)

FEATURES_TEMPLATE = "data/processed/features_{run_id}.parquet"
ALERTS_TEMPLATE = "data/processed/alerts_{run_id}.parquet"

# Parámetros del experimento de MTTD
REPEATS = 5
MAX_WAIT_SECONDS = 180  # timeout por repetición
POLL_INTERVAL = 5  # cada 5s revisa si ya hay alertas

REPORT_PATH = "data/processed/mttd_report.json"


def wait_for_alerts(alerts_path: str, timeout: int):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(alerts_path):
            df = pd.read_parquet(alerts_path)
            if len(df) > 0:
                return df, time.time()
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"No alerts generated within {timeout}s: {alerts_path}")


def run_build_alerts(run_id: str):
    features_path = FEATURES_TEMPLATE.format(run_id=run_id)
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"No existe features para run_id={run_id}: {features_path}. "
            "Primero ejecuta el paso que genera features_*.parquet."
        )

    cmd = BUILD_ALERTS_CMD.format(features=features_path, run_id=run_id)
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError(f"build_alerts falló con exit code={code}")


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def inject_fault_errors(rate=0.8, duration=60):
    url = "http://localhost:8102/fault/errors"
    r = requests.post(url, params={"rate": rate, "duration": duration}, timeout=10)
    assert r.status_code == 200, f"Fault injection failed: {r.status_code} {r.text}"
    return url, {"rate": rate, "duration": duration}


def test_mttd_end_to_end():
    results = []

    for i in range(REPEATS):
        run_id = "20260120T024536Z"
        alerts_path = ALERTS_TEMPLATE.format(run_id=run_id)

        # Limpieza por si existe (no debería)
        if os.path.exists(alerts_path):
            os.remove(alerts_path)

        # 1) Inyectar fault y marcar t0
        t0 = time.time()
        fault_url, fault_params = inject_fault_errors(rate=0.8, duration=60)
        print(f"[fault] {fault_url} params={fault_params}")

        # 2) Ejecutar construcción de alerts para ese run_id
        run_build_alerts(run_id)

        # 3) Esperar a que se materialice la primera alerta y marcar t1
        _, t1 = wait_for_alerts(alerts_path, MAX_WAIT_SECONDS)

        mttd = t1 - t0
        print(f"[MTTD] run_id={run_id} -> {mttd:.2f}s")
        results.append({"run_id": run_id, "mttd_seconds": mttd})

        # Pequeña pausa para evitar que todo caiga en la misma ventana
        time.sleep(2)

    # 4) Estadísticos
    mttds = [r["mttd_seconds"] for r in results]
    mean_mttd = sum(mttds) / len(mttds)

    report = {
        "repeats": REPEATS,
        "mean_mttd_seconds": mean_mttd,
        "p50_mttd_seconds": percentile(mttds, 50),
        "p95_mttd_seconds": percentile(mttds, 95),
        "samples": results,
    }

    os.makedirs("data/processed", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"[MTTD] mean={report['mean_mttd_seconds']:.2f}s "
        f"p50={report['p50_mttd_seconds']:.2f}s "
        f"p95={report['p95_mttd_seconds']:.2f}s"
    )
    print(f"[ok] saved report: {REPORT_PATH}")

    # Aserción de “latencia aceptable” usando el promedio (puedes cambiar a p95)
    assert mean_mttd <= MAX_WAIT_SECONDS, f"Mean MTTD too high: {mean_mttd:.2f}s"
