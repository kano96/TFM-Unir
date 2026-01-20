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
        description=(
            "Orquesta un experimento completo: tráfico -> fallos -> export métricas "
            "-> export logs -> features -> alerts -> grouping -> graph -> rca"
        )
    )
    parser.add_argument("--duration", type=int, default=300, help="Duración total (s).")
    parser.add_argument("--rps", type=float, default=1.0, help="Requests/sec global.")
    parser.add_argument("--services", default="user,auth,orders")
    parser.add_argument("--pad", type=int, default=30, help="Pad seconds para exports.")
    parser.add_argument(
        "--limit", type=int, default=5000, help="Líneas máximas por query en Loki."
    )
    parser.add_argument(
        "--weighted", action="store_true", help="Distribución de tráfico realista."
    )
    parser.add_argument("--tfidf-max-features", type=int, default=50)

    # E2E extras
    parser.add_argument(
        "--with-rca", action="store_true", help="Ejecuta pipeline RCA completo."
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top K RCA suggestions.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    models = root / "models"
    rca_dir = models / "rca"

    traffic_py = scripts / "traffic_generator.py"
    faults_py = scripts / "inject_faults.py"
    metrics_py = scripts / "export_metrics.py"
    logs_py = scripts / "export_logs.py"
    features_py = models / "features.py"

    for p in [traffic_py, faults_py, metrics_py, logs_py, features_py]:
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

    # 2) Fallos
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

    # 5) Features (alineadas al último run_id en labels)
    get_last_run_id = [
        sys.executable,
        "-c",
        """import pandas as pd;
        df=pd.read_csv('data/labels/fault_windows.csv');
        print(df['run_id'].astype(str).sort_values().iloc[-1])""",
    ]

    print("\n[run] starting end-to-end experiment")
    print(f"[run] services={args.services} duration={args.duration}s rps={args.rps}")

    # Ejecuta fase 1: generar datos raw
    run(traffic_cmd)
    run(faults_cmd)
    run(metrics_cmd)
    run(logs_cmd)

    # Resuelve run_id final desde labels
    print("\n[run] resolving last run_id from labels...")
    proc = subprocess.run(get_last_run_id, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    run_id = proc.stdout.strip()
    if not run_id:
        raise RuntimeError(
            "No se pudo resolver run_id desde data/labels/fault_windows.csv"
        )
    print(f"[run] run_id={run_id}")

    # Ejecuta features
    features_cmd = [
        sys.executable,
        str(features_py),
        "--run-id",
        run_id,
        "--metrics-dir",
        "data/raw/metrics",
        "--logs-dir",
        "data/raw/logs",
        "--out-dir",
        "data/processed",
        "--tfidf-max-features",
        str(args.tfidf_max_features),
    ]
    run(features_cmd)

    # Pipeline RCA opcional
    if args.with_rca:
        build_alerts_py = rca_dir / "build_alerts.py"
        group_alerts_py = rca_dir / "group_alerts.py"
        build_graph_py = rca_dir / "build_service_graph.py"
        rca_py = rca_dir / "rca.py"

        for p in [build_alerts_py, group_alerts_py, build_graph_py, rca_py]:
            if not p.exists():
                raise FileNotFoundError(f"No existe: {p}")

        alerts_out = f"data/processed/alerts_{run_id}.parquet"
        incidents_out = f"data/processed/incidents_{run_id}.parquet"
        graph_out = f"data/processed/service_graph_{run_id}.json"

        # build alerts
        run(
            [
                sys.executable,
                str(build_alerts_py),
                "--features",
                f"data/processed/features_{run_id}.parquet",
                "--out",
                alerts_out,
            ]
        )

        # group alerts -> incidents
        run(
            [
                sys.executable,
                str(group_alerts_py),
                "--alerts",
                alerts_out,
                "--features",
                f"data/processed/features_{run_id}.parquet",
                "--out-dir",
                "data/processed",
                "--eps",
                "0.8",
                "--min-samples",
                "3",
                "--time-weight",
                "0.05",
            ]
        )

        # build service graph
        run(
            [
                sys.executable,
                str(build_graph_py),
                "--out",
                graph_out,
            ]
        )

        # rca (elige el primer incidente por defecto)
        # si tu archivo incidents tiene ids tipo <run_id>_inc0, esto funcionará:
        incident_id = f"{run_id}_inc0"
        run(
            [
                sys.executable,
                str(rca_py),
                "--incidents",
                incidents_out,
                "--graph",
                graph_out,
                "--incident-id",
                incident_id,
                "--out-dir",
                "models/rca/out",
                "--top-k",
                str(args.top_k),
            ]
        )

    print("\n[run] ✅ experiment finished")
    print("[run] Outputs:")
    print("  - data/labels/fault_windows.csv")
    print(f"  - data/raw/metrics/metrics_{run_id}.*")
    print(f"  - data/raw/logs/logs_{run_id}.jsonl")
    print(f"  - data/processed/features_{run_id}.parquet")
    if args.with_rca:
        print(f"  - data/processed/alerts_{run_id}.parquet")
        print(f"  - data/processed/incidents_{run_id}.parquet")
        print(f"  - data/processed/service_graph_{run_id}.json")
        print(f"  - models/rca/out/rca_{incident_id}.json")


if __name__ == "__main__":
    main()
