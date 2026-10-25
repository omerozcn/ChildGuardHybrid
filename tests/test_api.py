import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tiny_model, monkeypatch):
    from api import main
    from api.services import model_service

    monkeypatch.setattr(model_service, "load", lambda: True)
    monkeypatch.setattr(model_service, "_model", tiny_model)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze(client):
    r = client.post("/api/analyze", json={"text": "kys you pathetic loser"})
    assert r.status_code == 200
    body = r.json()
    assert body["harmful"] is True
    assert {c["name"] for c in body["categories"]} == {
        "insult", "threat", "sexual", "identity_hate", "profanity"}
    assert body["targets_minor"]["name"] == "targets_minor"
    assert body["rule_hits"]


def test_analyze_rejects_empty(client):
    assert client.post("/api/analyze", json={"text": "   "}).status_code == 422
    assert client.post("/api/analyze", json={"text": ""}).status_code == 422


def test_batch(client):
    r = client.post("/api/analyze/batch", json={"texts": ["hi friend", "you idiot"]})
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_explain(client):
    r = client.post("/api/explain", json={"text": "you are a loser", "label": "insult"})
    assert r.status_code == 200
    assert len(r.json()["words"]) == 4
    assert client.post("/api/explain", json={"text": "x", "label": "nope"}).status_code == 422


def test_degraded_when_model_missing(monkeypatch):
    from api import main
    from api.services import model_service

    monkeypatch.setattr(model_service, "load", lambda: False)
    monkeypatch.setattr(model_service, "_model", None)
    monkeypatch.setattr(model_service, "_error", "Model bulunamadı")
    with TestClient(main.app) as c:
        assert c.get("/api/health").json()["status"] == "degraded"
        assert c.post("/api/analyze", json={"text": "hello"}).status_code == 503
