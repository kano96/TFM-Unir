from fastapi.testclient import TestClient
from services.simulator.main import app

client = TestClient(app)


def test_simulate_request_ok_or_error():
    """Debe responder correctamente con estado ok o 
    error y contener latencia"""
    response = client.get("/simulate")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "latency" in data
    assert 0.05 <= data["latency"] <= 0.5
    assert data["status"] in ["ok", "error"]


def test_metrics_endpoint_returns_prometheus_format():
    """El endpoint /metrics debe devolver texto con formato Prometheus"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert isinstance(response.text, str)
    assert "app_requests_total" in response.text
    assert "app_request_latency_seconds" in response.text
