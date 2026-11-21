"""FastAPI app: upload PDFs, then ask questions about them."""

import os
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config, db, embeddings, llm
from app.ingest import chunk_pdf

app = FastAPI(
    title="PDF Q&A (RAG)",
    description="Ask questions about your PDFs. PostgreSQL/pgvector for search, Claude for answers.",
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: int | None = None  # limit the search to one document


class Source(BaseModel):
    id: int
    filename: str
    page: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_id: int | None = None
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


class Passage(BaseModel):
    id: int
    filename: str
    page: int
    score: float
    content: str


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "PDF is larger than 20 MB.")

    try:
        chunks = chunk_pdf(data, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    except Exception as exc:  # pypdf raises several error types
        raise HTTPException(400, f"Could not read PDF: {exc}") from exc
    if not chunks:
        raise HTTPException(
            422, "No text found. Scanned PDFs (images only) need OCR first."
        )

    vectors = embeddings.embed([c.content for c in chunks])
    doc_id = db.save_document(file.filename, chunks, vectors)
    return {"id": doc_id, "filename": file.filename, "chunks": len(chunks)}


@app.get("/documents")
def list_documents():
    return db.list_documents()


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int):
    if not db.delete_document(doc_id):
        raise HTTPException(404, "Document not found.")


@app.post("/search", response_model=list[Passage])
def search(req: SearchRequest):
    """Retrieval only: the nearest chunks, no Claude call and no API key needed.

    The MCP server uses this so the client's own model writes the answer.
    """
    hits = db.search(embeddings.embed_one(req.query), req.top_k, req.document_id)
    return [
        Passage(id=i, filename=h.filename, page=h.page, score=round(h.score, 3), content=h.content)
        for i, h in enumerate(hits, start=1)
    ]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    hits = db.search(embeddings.embed_one(req.question), config.TOP_K, req.document_id)
    try:
        text = llm.answer(req.question, hits)
    except anthropic.AuthenticationError as exc:
        raise HTTPException(500, "ANTHROPIC_API_KEY is missing or invalid.") from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(429, "Claude rate limit reached, try again shortly.") from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(502, f"Claude API error: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(502, "Could not reach the Claude API.") from exc

    return AskResponse(
        answer=text,
        sources=[
            Source(
                id=i,
                filename=h.filename,
                page=h.page,
                score=round(h.score, 3),
                excerpt=h.content[:300],
            )
            for i, h in enumerate(hits, start=1)
        ],
    )
