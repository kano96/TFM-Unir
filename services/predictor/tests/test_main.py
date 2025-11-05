from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict_returns_200_and_valid_value():
    """Verifica que el endpoint responda 200 y
    devuelva una predicción numérica."""
    response = client.post("/predict", json={"values": [10.0, 12.5, 13.2, 14.8, 15.0]})
    assert response.status_code == 200
    data = response.json()
    assert "next_prediction" in data
    assert isinstance(data["next_prediction"], float)


def test_predict_increases_with_positive_trend():
    """Comprueba que con una tendencia ascendente la predicción
    sea mayor que el último valor."""
    values = [1, 2, 3, 4, 5, 6]
    response = client.post("/predict", json={"values": values})
    result = response.json()
    assert result["next_prediction"] >= values[-1]


def test_predict_with_constant_values():
    """Prueba con valores constantes: la predicción no debería variar mucho."""
    values = [5, 5, 5, 5, 5]
    response = client.post("/predict", json={"values": values})
    result = response.json()
    assert abs(result["next_prediction"] - 5) < 1.0


def test_predict_with_few_values():
    """Debe manejar correctamente series muy cortas."""
    values = [10.0, 11.0]
    response = client.post("/predict", json={"values": values})
    assert response.status_code == 200
    assert "next_prediction" in response.json()


def test_predict_with_empty_list():
    """Debe devolver error 422 (validación FastAPI) si la lista está vacía."""
    response = client.post("/predict", json={"values": []})
    assert response.status_code == 422
