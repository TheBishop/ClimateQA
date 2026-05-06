import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_ask_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400

def test_ask_returns_expected_fields():
    mock_result = {
        "answer": "Sea levels are projected to rise significantly.",
        "sources": [{"file": "IPCC.pdf", "page": 5, "snippet": "..."}]
    }
    with patch("src.api.main.ask", return_value=mock_result):
        response = client.post("/ask", json={"question": "What about sea level rise?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data
    assert data["question"] == "What about sea level rise?"
