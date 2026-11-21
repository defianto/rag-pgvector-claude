"""MCP server tests. A fake HTTP transport stands in for the FastAPI app,
so these need neither Docker nor an API key.

Run from the repo root:  python -m pytest -q mcp_server/tests
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx
from mcp import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import build_server  # noqa: E402


def run(handler, tool: str, args: dict):
    """Call one tool through a real MCP client connected in-process."""

    async def go():
        server = build_server(httpx.MockTransport(handler))
        async with Client(server) as client:
            return await client.call_tool(tool, args)

    return asyncio.run(go())


def text(result) -> str:
    return " ".join(c.text for c in result.content if c.type == "text")


def test_tools_are_listed_with_read_only_hints():
    async def go():
        async with Client(build_server()) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == {"list_documents", "search_documents", "ask_documents", "upload_pdf"}
    assert tools["search_documents"].annotations.read_only_hint is True
    assert tools["upload_pdf"].annotations.read_only_hint is False


def test_search_posts_query_and_returns_passages():
    seen = {}

    def handler(req: httpx.Request):
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=[
            {"id": 1, "filename": "hr.pdf", "page": 4, "score": 0.82, "content": "12 days of leave"}
        ])

    result = run(handler, "search_documents", {"query": "annual leave", "document_id": 7})
    assert not result.is_error
    assert seen == {"path": "/search", "body": {"query": "annual leave", "document_id": 7, "top_k": 5}}
    assert result.structured_content["result"][0]["page"] == 4


def test_search_rejects_bad_top_k_without_calling_api():
    def handler(req):
        raise AssertionError("API must not be called")

    result = run(handler, "search_documents", {"query": "x", "top_k": 50})
    assert result.is_error
    assert "top_k" in text(result)


def test_api_error_detail_is_passed_to_the_model():
    def handler(req):
        return httpx.Response(500, json={"detail": "ANTHROPIC_API_KEY is not set."})

    result = run(handler, "ask_documents", {"question": "hi"})
    assert result.is_error
    assert "ANTHROPIC_API_KEY is not set." in text(result)


def test_api_down_gives_a_helpful_message():
    def handler(req):
        raise httpx.ConnectError("connection refused")

    result = run(handler, "list_documents", {})
    assert result.is_error
    assert "docker compose up" in text(result)


def test_upload_sends_the_pdf(tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    def handler(req: httpx.Request):
        assert req.url.path == "/documents"
        assert b'filename="guide.pdf"' in req.content
        return httpx.Response(201, json={"id": 3, "filename": "guide.pdf", "chunks": 9})

    result = run(handler, "upload_pdf", {"path": str(pdf)})
    assert not result.is_error
    assert result.structured_content == {"id": 3, "filename": "guide.pdf", "chunks": 9}


def test_upload_rejects_non_pdf_and_missing_files(tmp_path):
    def handler(req):
        raise AssertionError("API must not be called")

    assert run(handler, "upload_pdf", {"path": str(tmp_path / "notes.txt")}).is_error
    assert run(handler, "upload_pdf", {"path": str(tmp_path / "missing.pdf")}).is_error
