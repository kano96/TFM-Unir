import argparse
import json
import os
import time
import random
from typing import Dict, List, Tuple, Optional

import requests

DEFAULT_PREDICTOR_URL = "http://localhost:8001/predict"
DEFAULT_HEALTH_URL = "http://localhost:8001/health"
DEFAULT_SERVICES = ["auth", "user", "orders"]


def get_expected_dim(health_url: str) -> int:
    r = requests.get(health_url, timeout=5)
    r.raise_for_status()
    data = r.json()
    if not data.get("model_loaded"):
        raise RuntimeError(f"Model not loaded. /health -> {data}")
    n = data.get("n_features_expected")
    return int(n) if n is not None else 110


def predict(predictor_url: str, service: str, feats: List[float]) -> Dict:
    payload = {"service": service, "features": feats}
    r = requests.post(predictor_url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def random_vec(dim: int, scale: float) -> List[float]:
    return [random.gauss(0.0, scale) for _ in range(dim)]


def clamp_vec(v: List[float], lo: float = -1.0, hi: float = 1.0) -> List[float]:
    return [max(lo, min(hi, x)) for x in v]


def jitter(v: List[float], sigma: float) -> List[float]:
    return clamp_vec([x + random.gauss(0.0, sigma) for x in v], lo=-1.0, hi=1.0)


def find_good_seed(
    predictor_url: str,
    dim: int,
    service: str,
    target_min: float,
    target_max: float,
    init_scale: float,
    min_scale: float,
    max_scale: float,
    attempts: int = 250,
) -> Tuple[List[float], float]:
    scale = init_scale
    best: Optional[Tuple[List[float], float]] = None
    best_dist = 10**9

    for _ in range(attempts):
        feats = clamp_vec(random_vec(dim, scale), lo=-1.0, hi=1.0)
        out = predict(predictor_url, service, feats)
        p = float(out["probability"])

        if target_min <= p <= target_max:
            return feats, p

        # distancia a la banda objetivo
        if p < target_min:
            dist = target_min - p
        else:
            dist = p - target_max

        if dist < best_dist:
            best_dist = dist
            best = (feats, p)

        # ajuste de escala según saturación
        if p >= 0.999:
            scale = max(min_scale, scale / 2.0)
        elif p <= 0.001:
            scale = min(max_scale, scale * 2.0)
        else:
            scale = max(min_scale, min(max_scale, scale * 0.9))

    if best is None:
        raise RuntimeError("Unable to find any seed (unexpected).")
    return best


def load_seeds(path: str) -> Optional[Dict[str, List[float]]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # formato esperado: {"dim": 110, "seeds": {"auth": [...], ...}}
    seeds = data.get("seeds")
    if not isinstance(seeds, dict):
        return None
    return {k: list(v) for k, v in seeds.items()}


def save_seeds(path: str, dim: int, seeds: Dict[str, List[float]]) -> None:
    payload = {"dim": dim, "seeds": seeds}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic predictor traffic for Grafana/Prometheus."
    )
    p.add_argument("--predictor-url", default=DEFAULT_PREDICTOR_URL)
    p.add_argument("--health-url", default=DEFAULT_HEALTH_URL)
    p.add_argument(
        "--services",
        default=",".join(DEFAULT_SERVICES),
        help="Comma-separated, e.g., auth,user,orders",
    )
    p.add_argument("--seeds-file", default="seeds.json")
    p.add_argument("--minutes", type=float, default=0.0, help="0 = run forever")
    p.add_argument(
        "--rps",
        type=float,
        default=1.5,
        help="Requests per second per service (approx.)",
    )

    p.add_argument("--target-min", type=float, default=0.20)
    p.add_argument("--target-max", type=float, default=0.80)
    p.add_argument("--sigma", type=float, default=0.02)

    p.add_argument("--init-scale", type=float, default=0.05)
    p.add_argument("--min-scale", type=float, default=1e-4)
    p.add_argument("--max-scale", type=float, default=0.5)

    p.add_argument(
        "--reseed",
        action="store_true",
        help="Ignore existing seeds file and re-search seeds",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    services = [s.strip() for s in args.services.split(",") if s.strip()]

    dim = get_expected_dim(args.health_url)
    print(f"[info] dim={dim} model ok. services={services}")

    seeds = None if args.reseed else load_seeds(args.seeds_file)
    if seeds:
        # verifica que existan todos
        missing = [s for s in services if s not in seeds]
        if missing:
            print(f"[warn] seeds file missing: {missing}. Will (re)compute those.")
        else:
            print(f"[ok] loaded seeds from {args.seeds_file}")

    if seeds is None:
        seeds = {}

    # Calcular seeds faltantes
    for svc in services:
        if svc in seeds and len(seeds[svc]) == dim:
            continue
        feats, p = find_good_seed(
            predictor_url=args.predictor_url,
            dim=dim,
            service=svc,
            target_min=args.target_min,
            target_max=args.target_max,
            init_scale=args.init_scale,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )
        seeds[svc] = feats
        print(f"[seed] {svc}: prob~{p:.3f}")

    save_seeds(args.seeds_file, dim, seeds)
    print(f"[ok] saved seeds -> {args.seeds_file}")

    # Timing: rps por servicio
    # delay aproximado por request de cada servicio:
    per_service_delay = max(0.01, 1.0 / max(0.1, args.rps))

    stop_at = None
    if args.minutes and args.minutes > 0:
        stop_at = time.time() + args.minutes * 60.0
        print(f"[info] will stop after {args.minutes} minutes")
    else:
        print("[info] running forever (CTRL+C to stop)")

    try:
        while True:
            for svc in services:
                feats = jitter(seeds[svc], sigma=args.sigma)
                out = predict(args.predictor_url, svc, feats)
                p = float(out["probability"])
                flag = bool(out["will_incident_within_horizon"])
                print(f"[{svc}] prob={p:.3f} flag={flag}")
                time.sleep(per_service_delay)

            if stop_at and time.time() >= stop_at:
                print("[ok] finished.")
                break
    except KeyboardInterrupt:
        print("\n[ok] stopped by user.")
