from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_detect_returns_200_and_valid_json():
    """Verifica que el endpoint /detect responda
    con 200 y devuelva un JSON válido."""
    response = client.post("/detect", json={"values": [0.1, 0.2, 0.3, 0.4]})
    assert response.status_code == 200
    result = response.json()
    assert "score" in result
    assert "is_anomaly" in result
    assert isinstance(result["score"], float)
    assert isinstance(result["is_anomaly"], bool)


def test_detect_identifies_extreme_values_as_anomalies():
    """Prueba que valores extremos generen una predicción de anomalía."""
    # valores normales
    normal_response = client.post(
        "/detect", json={"values": [0.1, 0.2, 0.3, 0.4]})
    normal_score = normal_response.json()["score"]

    # valores muy fuera de rango
    extreme_response = client.post(
        "/detect", json={"values": [50.0, 60.0, 70.0, 80.0]})
    extreme_result = extreme_response.json()

    assert extreme_response.status_code == 200
    # Esperamos que al menos los valores extremos sean más "anómalos"
    assert extreme_result["is_anomaly"] in [True, False]  # tipo correcto
    assert extreme_result["score"] <= normal_score or extreme_result["is_anomaly"] is True


def test_detect_with_empty_input_should_fail():
    """Debe devolver error 422 si no se proporcionan valores."""
    response = client.post("/detect", json={"values": []})
    assert response.status_code == 422


def test_detect_with_incorrect_type():
    """Debe devolver error 422 si los valores no son numéricos."""
    response = client.post("/detect", json={"values": ["a", "b", "c"]})
    assert response.status_code == 422
