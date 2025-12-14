import argparse
import random
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

import requests

DEFAULT_SERVICES = {
    "user": "http://localhost:8101",
    "auth": "http://localhost:8102",
    "orders": "http://localhost:8103",
}

DEFAULT_TIMEOUT = 3.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_services_arg(services_arg: str) -> List[str]:
    services = [s.strip() for s in services_arg.split(",") if s.strip()]
    if not services:
        raise ValueError("Debe indicar al menos un servicio en --services")
    return services


def safe_get(url: str, timeout: float) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="""Genera tráfico controlado hacia simulators
        (/simulate) para producir métricas/logs/trazas."""
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duración total en segundos (default: 300).",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=1.0,
        help="Requests por segundo globales (default: 1.0).",
    )
    parser.add_argument(
        "--services",
        default="user,auth,orders",
        help="Servicios objetivo separados por coma (default: user,auth,orders).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout HTTP por request en segundos (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.15,
        help="""Jitter aleatorio (0..1) sobre el sleep
        entre requests para evitar patrones perfectos (default: 0.15).""",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="""Si se habilita, distribuye el tráfico con pesos
        tipo realista (orders < user/auth).""",
    )
    args = parser.parse_args()

    selected = parse_services_arg(args.services)

    # Build target map
    targets: Dict[str, str] = {}
    for name in selected:
        if name not in DEFAULT_SERVICES:
            raise ValueError(
                f"""Servicio '{name}' no reconocido.
                Opciones: {sorted(DEFAULT_SERVICES.keys())}"""
            )
        targets[name] = DEFAULT_SERVICES[name]

    # Weights
    if args.weighted:
        # patrón típico: user/auth más tráfico que orders
        weights = {
            "user": 0.45,
            "auth": 0.35,
            "orders": 0.20,
        }
        w = [weights.get(s, 1.0) for s in targets.keys()]
    else:
        w = [1.0 for _ in targets.keys()]

    stop = {"flag": False}

    def handle_sigint(_sig, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    interval = 1.0 / max(args.rps, 0.0001)

    # Health check
    print(f"[traffic] start={utc_now_iso()} duration={args.duration}s rps={args.rps}")
    print(f"[traffic] targets={ {k: v for k, v in targets.items()} }")

    for svc, base in targets.items():
        ok = safe_get(f"{base}/health", timeout=args.timeout)
        if not ok:
            print(f"[warn] healthcheck failed for {svc} -> {base}/health")

    start_ts = time.time()
    end_ts = start_ts + args.duration

    sent = 0
    ok_count = 0
    err_count = 0

    svc_names = list(targets.keys())
    svc_bases = list(targets.values())

    while time.time() < end_ts and not stop["flag"]:
        # Choose a service
        idx = random.choices(range(len(svc_names)), weights=w, k=1)[0]
        svc = svc_names[idx]
        base = svc_bases[idx]

        url = f"{base}/simulate"
        t0 = time.time()
        try:
            resp = requests.get(url, timeout=args.timeout)
            sent += 1
            if resp.status_code == 200:
                ok_count += 1
            else:
                err_count += 1
        except requests.RequestException:
            sent += 1
            err_count += 1

        # pacing global
        elapsed = time.time() - t0
        sleep_for = max(interval - elapsed, 0.0)

        # jitter (± jitter%)
        if args.jitter > 0:
            jitter_factor = 1.0 + random.uniform(-args.jitter, args.jitter)
            sleep_for *= max(jitter_factor, 0.0)

        time.sleep(sleep_for)

        # pequeño log cada ~10 requests
        if sent % 10 == 0:
            print(
                f"""[traffic] t={int(time.time()-start_ts)}s
                sent={sent} ok={ok_count} err={err_count}"""
            )

    print(f"[traffic] done={utc_now_iso()} sent={sent} ok={ok_count} err={err_count}")
    if stop["flag"]:
        print("[traffic] stopped by user (SIGINT)")

    # exit code si todo falló
    if sent > 0 and ok_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
