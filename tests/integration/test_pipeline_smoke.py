import requests

SERVICES = {
    "sim_user": "http://localhost:8101/health",
    "sim_auth": "http://localhost:8102/health",
    "sim_orders": "http://localhost:8103/health",
    "prom": "http://localhost:9090/-/healthy",
    "loki": "http://localhost:3100/ready",
    "detector": "http://localhost:8000/health",
    "predictor": "http://localhost:8001/health",
    "rca": "http://localhost:8002/health",
}


def test_health_all():
    for name, url in SERVICES.items():
        r = requests.get(url, timeout=10)
        assert r.status_code == 200, f"{name} failed: {r.status_code} {r.text}"
