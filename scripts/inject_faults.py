import csv
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests


# -----------------------------
# Config
# -----------------------------
SIMULATORS: Dict[str, str] = {
    "user": "http://localhost:8101",
    "auth": "http://localhost:8102",
    "orders": "http://localhost:8103",
}

LABELS_PATH = os.getenv("LABELS_PATH", "data/labels/fault_windows.csv")
RUN_SECONDS = int(os.getenv("RUN_SECONDS", "300"))  # duración total del experimento
# requests por segundo por servicio
TRAFFIC_RPS = float(os.getenv("TRAFFIC_RPS", "1.0"))

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3.0"))


@dataclass
class FaultEvent:
    # cuándo iniciar desde t=0 (segundos)
    at: int
    # duración del fault en segundos
    duration: int
    # servicio lógico (key SIMULATORS)
    service: str
    # tipo de fault: latency/errors/cpu/memory
    fault_type: str
    # parámetros específicos del fault
    params: Dict[str, str]


# -----------------------------
# Plan de fallos
# -----------------------------
FAULT_SCHEDULE: List[FaultEvent] = [
    # 1) Latencia en orders desde segundo 30 por 60s
    FaultEvent(
        at=30,
        duration=60,
        service="orders",
        fault_type="latency",
        params={"ms": "400"},
    ),
    # 2) Errores altos en auth desde segundo 120 por 45s
    FaultEvent(
        at=120,
        duration=45,
        service="auth",
        fault_type="errors",
        params={"rate": "0.6"},
    ),
    # 3) Spike de CPU en user desde segundo 200 por 20s
    FaultEvent(
        at=200,
        duration=20,
        service="user",
        fault_type="cpu",
        params={},
    ),
]


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dirs(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def post(url: str, params: Optional[dict] = None) -> requests.Response:
    return requests.post(url, params=params or {}, timeout=REQUEST_TIMEOUT)


def get(url: str) -> requests.Response:
    return requests.get(url, timeout=REQUEST_TIMEOUT)


# -----------------------------
# Traffic generator
# -----------------------------
def traffic_worker(
    service_name: str, base_url: str, stop_event: threading.Event
) -> None:
    interval = 1.0 / TRAFFIC_RPS if TRAFFIC_RPS > 0 else 0.0

    # warmup
    try:
        get(f"{base_url}/health")
    except Exception:
        pass

    while not stop_event.is_set():
        start = time.time()
        try:
            r = get(f"{base_url}/simulate")
            # opcional: print mínimo para debug
            if r.status_code != 200:
                print(f"[traffic] {service_name} status={r.status_code}")
        except Exception as e:
            print(f"[traffic] {service_name} error={e}")

        # control de tasa
        elapsed = time.time() - start
        sleep_for = max(0.0, interval - elapsed)
        time.sleep(sleep_for)


# -----------------------------
# Fault injector + labeling
# -----------------------------
def inject_fault(event: FaultEvent) -> Dict[str, str]:
    base_url = SIMULATORS[event.service]
    endpoint = f"{base_url}/fault/{event.fault_type}"

    # añadimos duration como query param para los faults que lo soportan
    params = dict(event.params)
    params["duration"] = str(event.duration)

    # CPU/memory también usan duration; latency/errors usan duration
    resp = post(endpoint, params=params)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Fault injection failed: service={event.service} "
            f"type={event.fault_type} status={resp.status_code} body={resp.text}"
        )

    return {
        "service": event.service,
        "fault_type": event.fault_type,
        "start_ts": now_iso(),
        # end_ts calculado por duración (en UTC)
        "end_ts": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .astimezone(timezone.utc)
        .isoformat(timespec="seconds"),
    }


def write_labels_csv(rows: List[Dict[str, str]], path: str) -> None:
    ensure_dirs(path)

    needs_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_id", "service", "fault_type", "start_ts", "end_ts"]
        )
        if needs_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def compute_end_ts_from_start(start_iso: str, duration_s: int) -> str:
    # start_iso viene con tz UTC;
    # formato: 2025-12-14T00:00:00+00:00
    dt = datetime.fromisoformat(start_iso)
    end_dt = dt + timedelta(seconds=duration_s)
    return end_dt.isoformat(timespec="seconds")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # Sanity checks
    for name, url in SIMULATORS.items():
        try:
            r = get(f"{url}/health")
            if r.status_code != 200:
                raise RuntimeError(f"{name} unhealthy: {r.status_code}")
        except Exception as e:
            raise RuntimeError(
                f"Simulator '{name}' not reachable at {url}. "
                f"Is docker compose up? Error: {e}"
            ) from e

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"[run] run_id={run_id} duration={RUN_SECONDS}s traffic_rps={TRAFFIC_RPS}")

    stop_event = threading.Event()
    threads: List[threading.Thread] = []

    # Start traffic workers
    for svc, base in SIMULATORS.items():
        t = threading.Thread(
            target=traffic_worker, args=(svc, base, stop_event), daemon=True
        )
        t.start()
        threads.append(t)

    # Execute schedule
    start_time = time.time()
    labels_buffer: List[Dict[str, str]] = []

    # Ordena eventos por "at"
    schedule = sorted(FAULT_SCHEDULE, key=lambda e: e.at)
    next_idx = 0

    try:
        while True:
            elapsed = int(time.time() - start_time)
            if elapsed >= RUN_SECONDS:
                break

            # Inyecta eventos cuando corresponde
            while next_idx < len(schedule) and elapsed >= schedule[next_idx].at:
                ev = schedule[next_idx]
                print(
                    f"[fault] t={elapsed}s inject service={ev.service} "
                    f"type={ev.fault_type} duration={ev.duration}s params={ev.params}"
                )

                start_ts = now_iso()
                # inyecta
                resp = post(
                    f"{SIMULATORS[ev.service]}/fault/{ev.fault_type}",
                    params={**ev.params, "duration": str(ev.duration)},
                )
                if resp.status_code != 200:
                    print(
                        f"[fault] FAILED service={ev.service} type={ev.fault_type} "
                        f"status={resp.status_code} body={resp.text}"
                    )
                else:
                    end_ts = datetime.fromisoformat(start_ts) + timedelta(
                        seconds=ev.duration
                    )
                    labels_buffer.append(
                        {
                            "run_id": run_id,
                            "service": ev.service,
                            "fault_type": ev.fault_type,
                            "start_ts": start_ts,
                            "end_ts": end_ts.isoformat(timespec="seconds"),
                        }
                    )
                    # escribe incrementalmente por seguridad
                    write_labels_csv([labels_buffer[-1]], LABELS_PATH)

                next_idx += 1

            time.sleep(1)

    finally:
        # Stop traffic
        stop_event.set()
        time.sleep(1)

        # Clear faults to leave env clean
        for svc, base in SIMULATORS.items():
            try:
                post(f"{base}/fault/clear")
            except Exception:
                pass

        print(f"[run] finished run_id={run_id}")
        print(f"[labels] saved to: {LABELS_PATH}")


if __name__ == "__main__":
    from datetime import timedelta  # local import for compute_end_ts

    main()
