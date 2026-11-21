# PDF Q&A with RAG (FastAPI + PostgreSQL/pgvector + Claude)

Upload PDFs, then ask questions about them in plain language. The app finds
the most relevant passages with vector search in PostgreSQL and asks Claude to
answer **only from those passages**, citing the file and page for each claim.

Works across languages: ask in Indonesian about an English PDF (or the other
way round) and it still finds the right passage.

## How it works

```
Upload:  PDF ──► extract text per page ──► split into overlapping chunks
               ──► embed each chunk (local model) ──► store in Postgres (pgvector)

Ask:     question ──► embed ──► top-k nearest chunks (cosine, HNSW index)
               ──► Claude answers from those chunks, with [n] citations
```

**RAG** (Retrieval-Augmented Generation) means the model looks up your own
documents first, then writes the answer from what it found. That keeps answers
grounded in your data and lets the app say "not in the documents" instead of
making something up.

| Part | Choice | Why |
|---|---|---|
| API | FastAPI | Typed request/response models, auto docs at `/docs` |
| Vector store | PostgreSQL 17 + pgvector, HNSW index | One database for data and vectors, plain SQL |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` via fastembed (384-dim, CPU) | Free, runs locally, multilingual |
| Answering | Claude API (`claude-opus-5-5`, low effort) | Strong at grounded, cited answers |
| PDF parsing | pypdf | Pure Python, no system deps |

## Run it

You need Docker and an Anthropic API key (https://console.anthropic.com).

```bash
cp .env.example .env        # then put your key in ANTHROPIC_API_KEY
docker compose up --build
```

Open http://localhost:8000, upload a PDF, ask a question.
API docs: http://localhost:8000/docs

If port 8000 or 5433 is already taken on your machine, set `API_PORT` or
`DB_PORT` in `.env`.

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/documents` | Upload a PDF (multipart field `file`) |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/documents/{id}` | Delete a document and its chunks |
| `POST` | `/ask` | `{"question": "...", "document_id": null}` → answer + sources |
| `POST` | `/search` | `{"query": "...", "document_id": null, "top_k": 5}` → passages only, no Claude call |

Example:

```bash
curl -F "file=@handbook.pdf" http://localhost:8000/documents
curl -H "Content-Type: application/json" \
     -d '{"question": "How many days of annual leave do I get?"}' \
     http://localhost:8000/ask
```

```json
{
  "answer": "Employees get 12 days of annual leave per year [1].",
  "sources": [
    {"id": 1, "filename": "handbook.pdf", "page": 4, "score": 0.82, "excerpt": "..."}
  ]
}
```

## Use it from Claude Desktop (MCP)

`mcp_server/` wraps this API as an MCP server, so Claude Desktop can list,
search and upload your PDFs as tools. Setup and tool list:
[mcp_server/README.md](mcp_server/README.md).

## Project layout

```
app/
  main.py        FastAPI routes
  ingest.py      PDF text extraction and chunking
  embeddings.py  local embedding model
  db.py          Postgres/pgvector queries
  llm.py         prompt and Claude call
  static/        minimal web UI
db/init.sql      schema + HNSW index
tests/           unit tests (chunking, prompt, /search)
mcp_server/      MCP server for Claude Desktop (own requirements and tests)
```

## Tests

```bash
docker compose run --rm -v "$PWD/tests:/code/tests" api \
  sh -c "pip install -q pytest && python -m pytest -q tests"
```

## Design notes

- **Chunking:** ~1000 characters with 200 overlap, cut at word boundaries, kept
  per page so every answer can cite a page number.
- **Grounding:** the prompt tells Claude to use only the numbered sources, cite
  them, and say so when the answer is not there.
- **Cost:** embeddings run locally, so uploading is free. Each question is one
  Claude call with only the top 5 chunks, not the whole PDF.
- **Limits:** scanned PDFs (images only) have no text layer and need OCR first.

## Possible next steps

- Hybrid search (pgvector + Postgres full-text) for exact terms like IDs
- Streaming answers to the UI
- An evaluation set of question/answer pairs to measure retrieval quality
