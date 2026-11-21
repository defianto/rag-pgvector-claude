"""MCP server that lets Claude Desktop (or any MCP client) use the PDF Q&A API.

It is a thin wrapper: every tool is one HTTP call to the FastAPI app, so the
app keeps owning the database, the embeddings and the Claude API key. This
process only needs `mcp` and `httpx`.

Run it with `python mcp_server/server.py`. It talks MCP over stdin/stdout, so
it must never print to stdout; anything for humans goes to stderr.

Settings (environment variables):
    RAG_API_URL      base URL of the FastAPI app (default http://localhost:8000)
    RAG_API_TIMEOUT  seconds to wait for one API call (default 120)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT = float(os.getenv("RAG_API_TIMEOUT", "120"))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # same limit as the API

logging.getLogger("httpx").setLevel(logging.WARNING)  # one log line per request is noise

INSTRUCTIONS = """\
Tools for the user's own PDF library (a RAG service running on their machine).

To answer a question about the PDFs, prefer `search_documents`: it returns the
most relevant passages with file name and page, and you write the answer from
them. Use only those passages, cite them like [1] or (handbook.pdf, p. 4), and
say plainly when they do not contain the answer. Call `list_documents` first
when the user names a document, so you can pass its `document_id`.

`ask_documents` asks the service's own Claude call for a finished answer. It
costs the user API credit, so use it only when they ask for it.
"""


class Document(BaseModel):
    id: int
    filename: str
    uploaded_at: str
    chunks: int


class Passage(BaseModel):
    id: int
    filename: str
    page: int
    score: float
    content: str


class Source(BaseModel):
    id: int
    filename: str
    page: int
    score: float
    excerpt: str


class Answer(BaseModel):
    answer: str
    sources: list[Source]


class Uploaded(BaseModel):
    id: int
    filename: str
    chunks: int


def build_server(transport: httpx.AsyncBaseTransport | None = None) -> MCPServer:
    """Create the server. Tests pass a fake `transport` instead of a live API."""

    mcp = MCPServer("rag-pdf", instructions=INSTRUCTIONS, version="0.1.0")

    async def call(method: str, path: str, **kwargs) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=API_URL, timeout=API_TIMEOUT, transport=transport
        ) as http:
            try:
                resp = await http.request(method, path, **kwargs)
            except httpx.ConnectError as exc:
                raise ToolError(
                    f"Cannot reach the RAG API at {API_URL}. "
                    "Start it with `docker compose up -d` and try again."
                ) from exc
            except httpx.TimeoutException as exc:
                raise ToolError(f"The RAG API did not answer within {API_TIMEOUT:.0f} s.") from exc
        if resp.is_error:
            raise ToolError(f"RAG API error {resp.status_code}: {_detail(resp)}")
        return resp

    @mcp.tool(
        title="List PDF documents",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def list_documents() -> list[Document]:
        """List the PDFs in the library: id, file name, upload time, number of chunks."""
        resp = await call("GET", "/documents")
        return [Document(**{**d, "uploaded_at": str(d["uploaded_at"])}) for d in resp.json()]

    @mcp.tool(
        title="Search PDF passages",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def search_documents(
        query: str, document_id: int | None = None, top_k: int = 5
    ) -> list[Passage]:
        """Find the passages most relevant to `query` (semantic search, any language).

        Returns up to `top_k` passages (1-20) with file name, page and a
        similarity score from 0 to 1. Set `document_id` to search one PDF only.
        Free: no LLM is called.
        """
        if not query.strip():
            raise ToolError("query must not be empty.")
        if not 1 <= top_k <= 20:
            raise ToolError("top_k must be between 1 and 20.")
        resp = await call(
            "POST", "/search",
            json={"query": query, "document_id": document_id, "top_k": top_k},
        )
        return [Passage(**p) for p in resp.json()]

    @mcp.tool(
        title="Ask the PDF library",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    )
    async def ask_documents(question: str, document_id: int | None = None) -> Answer:
        """Get a finished, cited answer from the RAG service's own Claude call.

        Uses the service's Anthropic API key, so each call costs credit.
        Prefer `search_documents` unless the user asks for this.
        """
        if not question.strip():
            raise ToolError("question must not be empty.")
        resp = await call(
            "POST", "/ask", json={"question": question, "document_id": document_id}
        )
        return Answer(**resp.json())

    @mcp.tool(
        title="Upload a PDF",
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
        ),
    )
    async def upload_pdf(path: str) -> Uploaded:
        """Add a PDF from this computer to the library, so it can be searched.

        `path` is a full path to a .pdf file, for example C:\\Users\\me\\handbook.pdf.
        Uploading the same file twice stores it twice.
        """
        file = Path(path).expanduser()
        if file.suffix.lower() != ".pdf":
            raise ToolError("Only .pdf files are supported.")
        if not file.is_file():
            raise ToolError(f"File not found: {file}")
        if file.stat().st_size > MAX_UPLOAD_BYTES:
            raise ToolError("PDF is larger than 20 MB.")
        resp = await call(
            "POST", "/documents",
            files={"file": (file.name, file.read_bytes(), "application/pdf")},
        )
        return Uploaded(**resp.json())

    return mcp


def _detail(resp: httpx.Response) -> str:
    """FastAPI puts the reason in {"detail": ...}; fall back to the raw text."""
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    return str(detail or resp.text or resp.reason_phrase)[:500]


mcp = build_server()

if __name__ == "__main__":
    mcp.run()  # stdio transport, which is what Claude Desktop uses
