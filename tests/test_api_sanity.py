import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root_healthcheck():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "SmartLoanScorer API is running" in resp.json().get("message", "") 