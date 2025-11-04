from main import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_rca_returns_possible_causes():
    """
    Verifica que el servicio RCA identifique correctamente los nodos raíz
    en un grafo de dependencias dirigido.
    """
    # Grafo: A → B → C, A → D
    payload = {
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "A", "target": "D"}
        ]
    }

    response = client.post("/rca", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "possible_causes" in data
    assert isinstance(data["possible_causes"], list)
    # A es el único nodo sin dependencias entrantes
    assert data["possible_causes"] == ["A"]


def test_rca_with_multiple_roots():
    """
    Verifica que el servicio maneje múltiples causas raíz
    (nodos sin dependencias entrantes).
    """
    payload = {
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "C", "target": "D"}
        ]
    }

    response = client.post("/rca", json=payload)
    assert response.status_code == 200

    data = response.json()
    # A y C son los nodos raíz
    assert set(data["possible_causes"]) == {"A", "C"}


def test_rca_with_empty_edges():
    """
    Verifica el comportamiento con un grafo vacío.
    """
    payload = {"edges": []}
    response = client.post("/rca", json=payload)
    assert response.status_code == 200
    data = response.json()
    # No hay causas raíz porque no existen nodos
    assert data["possible_causes"] == []


def test_rca_with_missing_field():
    """
    Verifica que un payload mal formado genere error 422 (Unprocessable Entity)
    por falta de campo requerido.
    """
    payload = {}  # Falta "edges"
    response = client.post("/rca", json=payload)
    assert response.status_code == 422
