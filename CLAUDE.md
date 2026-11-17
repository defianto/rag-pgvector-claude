# CLAUDE.md

Guidance for AI coding agents (Claude Code and others) working in this repo.
Read this before changing code. Humans may find it useful too.

## What this project is

PDF question answering with RAG. Users upload PDFs; text is chunked, embedded
locally, and stored in PostgreSQL/pgvector. Questions retrieve the nearest
chunks, and Claude answers from those chunks only, citing file and page.

## Architecture

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI routes: `POST/GET/DELETE /documents`, `POST /ask`, `/health`, serves the UI |
| `app/ingest.py` | PDF text extraction (pypdf) and chunking (`split_text`, `chunk_pdf`) |
| `app/embeddings.py` | Local embedding model via fastembed (384-dim, multilingual) |
| `app/db.py` | All SQL: save/list/delete documents, cosine search with `<=>` |
| `app/llm.py` | System prompt, `build_prompt`, the Claude API call |
| `app/config.py` | Every setting, read from environment variables |
| `app/static/index.html` | Single-file UI, plain JS, no build step |
| `db/init.sql` | Schema and HNSW index; runs only when the DB volume is first created |

Data flow: `upload -> ingest.chunk_pdf -> embeddings.embed -> db.save_document`
and `ask -> embeddings.embed_one -> db.search -> llm.answer`.

## Commands

All commands run from the repo root. The app runs in Docker; the host Python
version may not have wheels for every dependency, so prefer the container.

```bash
docker compose up -d --build          # start db + api (UI at http://localhost:${API_PORT:-8000})
docker compose logs -f api            # API logs
docker compose run --rm -v "$PWD/tests:/code/tests" api \
  sh -c "pip install -q pytest && python -m pytest -q tests"   # unit tests
docker compose down                   # stop (keeps data)
docker compose down -v                # stop AND delete the database volume
```

Health check: `curl http://localhost:${API_PORT:-8000}/health` returns `{"status":"ok"}`.

## Rules

- **Never commit `.env`** or print, log, or paste the value of `ANTHROPIC_API_KEY`.
  Put new settings in `app/config.py` with a safe default and document them,
  commented out, in `.env.example`.
- **Do not call the Claude API in tests.** It costs money and needs a key. Test
  pure functions (`split_text`, `build_prompt`) or mock `llm._get_client`.
- **Ask before** `docker compose down -v`, deleting documents through the API, or
  `git push`. These lose data or publish work.
- **Embedding dimension is fixed at 384.** Changing `EMBEDDING_MODEL` to one with
  a different size also requires changing `vector(384)` in `db/init.sql`,
  `EMBEDDING_DIM` in `app/config.py`, and recreating the DB volume.
- **Schema changes:** `db/init.sql` only runs on a fresh volume. For an existing
  database, write the `ALTER` statement too and say so in the PR.
- Keep SQL in `app/db.py` and Claude calls in `app/llm.py`. Routes in `main.py`
  stay thin: validate input, call those modules, map errors to HTTP codes.

## Conventions

- Python 3.12, type hints on public functions, small dataclasses for records
  (`Chunk`, `SearchHit`).
- Parameterised SQL only (`%s` / `%(name)s`); never build SQL from user input.
- Claude: use the official `anthropic` SDK. Model, effort and max tokens come
  from `config` (default `claude-opus-5-5`, effort `low`). Always check
  `stop_reason == "refusal"` before reading the text.
- Prompt changes: keep the "answer only from sources, cite [n], say when the
  answer is not there" behaviour. It is the core promise of the app.
- User-facing API errors are short English sentences in `HTTPException`.
- Commit messages: imperative mood, one-line summary under 72 characters.

## Workflow for agents

1. Read the files you will touch and their tests before editing.
2. Make the smallest change that solves the task; don't reformat unrelated code.
3. Add or update a unit test when you change chunking, prompting or SQL.
4. Run the tests in the container and confirm `/health` still answers after
   `docker compose up -d --build`.
5. For behaviour changes, try one upload and one question in the UI, and only
   use the API key with the owner's OK.
6. Update `README.md` when commands, settings or endpoints change.
7. Commit on a branch, summarise what changed and how it was verified, and
   wait for the owner before pushing.
