import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n[cmd] {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="""Orquesta un experimento completo: tráfico -> fallos
        -> export métricas -> export logs."""
    )
    parser.add_argument("--duration", type=int, default=300, help="Duración total (s).")
    parser.add_argument("--rps", type=float, default=1.0, help="Requests/sec global.")
    parser.add_argument(
        "--services",
        default="user,auth,orders",
        help="Servicios comma-separated: user,auth,orders",
    )

    parser.add_argument("--pad", type=int, default=30, help="Pad seconds para exports.")
    parser.add_argument(
        "--limit", type=int, default=5000, help="Líneas máximas por query en Loki."
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Distribución de tráfico tipo realista (user/auth > orders).",
    )

    args = parser.parse_args()

    # Paths
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"

    traffic_py = scripts / "traffic_generator.py"
    faults_py = scripts / "inject_faults.py"
    metrics_py = scripts / "export_metrics.py"
    logs_py = scripts / "export_logs.py"

    for p in [traffic_py, faults_py, metrics_py, logs_py]:
        if not p.exists():
            raise FileNotFoundError(f"No existe: {p}")

    # 1) Tráfico
    traffic_cmd = [
        sys.executable,
        str(traffic_py),
        "--duration",
        str(args.duration),
        "--rps",
        str(args.rps),
        "--services",
        args.services,
    ]
    if args.weighted:
        traffic_cmd.append("--weighted")

    # 2) Fallos (usa su lógica interna de schedule)
    faults_cmd = [sys.executable, str(faults_py)]

    # 3) Export métricas (alineadas al último run_id)
    metrics_cmd = [
        sys.executable,
        str(metrics_py),
        "--pad",
        str(args.pad),
        "--services",
        args.services,
    ]

    # 4) Export logs (alineados al último run_id)
    logs_cmd = [
        sys.executable,
        str(logs_py),
        "--pad",
        str(args.pad),
        "--limit",
        str(args.limit),
        "--services",
        args.services,
    ]

    print("\n[run] starting end-to-end experiment")
    print(f"[run] services={args.services} duration={args.duration}s rps={args.rps}")

    # Ejecuta
    run(traffic_cmd)
    run(faults_cmd)
    run(metrics_cmd)
    run(logs_cmd)

    print("\n[run] ✅ experiment finished")
    print("[run] Revisa outputs en:")
    print("  - data/labels/fault_windows.csv")
    print("  - data/raw/metrics/")
    print("  - data/raw/logs/")


if __name__ == "__main__":
    main()
