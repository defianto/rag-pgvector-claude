"""POST /search: retrieval only, used by the MCP server. DB and model are faked."""

import numpy as np
from fastapi.testclient import TestClient

from app import db, embeddings, main
from app.db import SearchHit


def test_search_returns_full_passages_without_calling_claude(monkeypatch):
    calls = {}

    def fake_search(vector, top_k, document_id):
        calls.update(top_k=top_k, document_id=document_id)
        return [SearchHit("hr.pdf", 4, "x" * 500, 0.81234)]

    monkeypatch.setattr(embeddings, "embed_one", lambda text: np.zeros(384))
    monkeypatch.setattr(db, "search", fake_search)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    resp = TestClient(main.app).post("/search", json={"query": "cuti", "top_k": 3})

    assert resp.status_code == 200
    assert resp.json() == [
        {"id": 1, "filename": "hr.pdf", "page": 4, "score": 0.812, "content": "x" * 500}
    ]
    assert calls == {"top_k": 3, "document_id": None}


def test_search_validates_input():
    client = TestClient(main.app)
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "a", "top_k": 99}).status_code == 422
