"""
validate_ingestion.py

Valida que el canal de ingesta de la Fase 2 esté operativo:
- Prometheus recoge métricas de los simulators
- Loki contiene logs de los simulators
- Jaeger está disponible y recibe trazas

Este script debe ejecutarse con el stack Docker Compose levantado.
"""

import sys
import requests

PROMETHEUS_URL = "http://localhost:9090"
LOKI_URL = "http://localhost:3100"
JAEGER_URL = "http://localhost:16686"

SIMULATOR_SERVICES = ["simulator-user", "simulator-auth", "simulator-orders"]
TIMEOUT = 10


# -------------------------
# Helpers
# -------------------------
def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


# -------------------------
# Prometheus validation
# -------------------------
def validate_prometheus() -> None:
    print("\n[CHECK] Prometheus metrics")

    try:
        resp = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        fail(f"Prometheus no accesible: {e}")

    data = resp.json()
    active_targets = data.get("data", {}).get("activeTargets", [])

    found = {s: False for s in SIMULATOR_SERVICES}

    for t in active_targets:
        instance = t.get("labels", {}).get("instance", "")
        health = t.get("health", "")

        for svc in SIMULATOR_SERVICES:
            if svc in instance and health == "up":
                found[svc] = True

    missing = [svc for svc, ok_ in found.items() if not ok_]
    if missing:
        fail(f"Prometheus no scrapea correctamente: {missing}")

    ok("Prometheus scrapea métricas de todos los simulators")


# -------------------------
# Loki validation
# -------------------------
def validate_loki() -> None:
    print("\n[CHECK] Loki logs")

    try:
        resp = requests.get(
            f"{LOKI_URL}/loki/api/v1/label/container/values", timeout=TIMEOUT
        )
        resp.raise_for_status()
    except Exception as e:
        fail(f"Loki no accesible: {e}")

    containers = resp.json().get("data", [])

    found = []
    for svc in SIMULATOR_SERVICES:
        if any(svc in c for c in containers):
            found.append(svc)

    missing = [svc for svc in SIMULATOR_SERVICES if svc not in found]
    if missing:
        fail(f"No se encontraron logs en Loki para: {missing}")

    ok("Loki contiene logs de los simulators")


# -------------------------
# Jaeger validation
# -------------------------
def validate_jaeger() -> None:
    print("\n[CHECK] Jaeger traces")

    try:
        resp = requests.get(f"{JAEGER_URL}/api/services", timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        fail(f"Jaeger no accesible: {e}")

    services = resp.json().get("data", [])

    found = []
    for svc in ["user", "auth", "orders"]:
        if svc in services:
            found.append(svc)

    if not found:
        fail("No se detectaron servicios con trazas en Jaeger")

    ok(f"Jaeger recibe trazas de servicios: {found}")


# -------------------------
# Main
# -------------------------
def main() -> None:
    print("=== Validación de Ingesta – Fase 2 ===")

    validate_prometheus()
    validate_loki()
    validate_jaeger()

    print("\n[SUCCESS] Canal de ingesta validado correctamente 🎉")


if __name__ == "__main__":
    main()
