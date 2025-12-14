from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    """El endpoint /health debe responder OK e incluir el nombre del servicio."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_simulate_request_ok_or_error():
    """Debe responder con status ok/error, incluir latencia y service.
    Nota: ahora la latencia puede superar 0.5 si hay fault de latencia activo,
    por lo que aquí validamos límites razonables en modo normal.
    """
    response = client.get("/simulate")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ["ok", "error"]
    assert "latency" in data
    assert isinstance(data["latency"], (int, float))
    assert 0.05 <= data["latency"] <= 0.55
    assert data["service"]


def test_fault_latency_increases_latency():
    """Al activar fault de latencia,
    la latencia debe aumentar respecto al mínimo base."""
    # Limpia cualquier fault previo
    clear_resp = client.post("/fault/clear")
    assert clear_resp.status_code == 200

    # Activa latencia extra de 300ms por 60s
    fault_resp = client.post("/fault/latency", params={"ms": 300, "duration": 60})
    assert fault_resp.status_code == 200
    fault_data = fault_resp.json()
    assert fault_data["fault"] == "latency"

    # Mide una simulación: debería ser >= 0.05 + 0.3 aprox
    response = client.get("/simulate")
    assert response.status_code == 200
    data = response.json()
    assert "latency" in data
    assert data["latency"] >= 0.30

    client.post("/fault/clear")


def test_fault_errors_increases_error_rate():
    """Con un error_rate alto, en varias llamadas debe aparecer al menos un 'error'."""
    client.post("/fault/clear")

    fault_resp = client.post("/fault/errors", params={"rate": 0.95, "duration": 60})
    assert fault_resp.status_code == 200
    fault_data = fault_resp.json()
    assert fault_data["fault"] == "errors"

    statuses = []
    for _ in range(10):
        r = client.get("/simulate")
        assert r.status_code == 200
        statuses.append(r.json()["status"])

    assert "error" in statuses

    client.post("/fault/clear")


def test_metrics_endpoint_returns_prometheus_format():
    """El endpoint /metrics debe devolver texto con
    formato Prometheus, incluyendo labels."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert isinstance(response.text, str)
    assert "app_requests_total" in response.text
    assert "app_request_latency_seconds" in response.text
    assert "app_errors_total" in response.text
    assert "app_fault_active" in response.text
