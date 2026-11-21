# MCP server: use your PDF library from Claude Desktop

This folder turns the RAG API into an [MCP](https://modelcontextprotocol.io)
server, so Claude Desktop (or any MCP client) can search your PDFs as a tool.

```
Claude Desktop ──stdio (MCP)──► mcp_server/server.py ──HTTP──► FastAPI app ──► Postgres/pgvector
```

The server is a thin wrapper: every tool is one HTTP call to the running API.
The database, embeddings and Anthropic key stay in the Docker app; this process
only needs `mcp` and `httpx`.

## Tools

| Tool | Calls | Cost | Notes |
|---|---|---|---|
| `list_documents` | `GET /documents` | free | read-only |
| `search_documents(query, document_id?, top_k=5)` | `POST /search` | free | returns passages; Claude Desktop writes the cited answer |
| `ask_documents(question, document_id?)` | `POST /ask` | API credit | the app's own Claude call, for a finished answer |
| `upload_pdf(path)` | `POST /documents` | free | reads a PDF from this computer |

There is deliberately no delete tool: deleting stays a human action in the web UI.
Each tool carries MCP annotations (`readOnlyHint`, `destructiveHint`, ...) so the
client can decide what needs the user's approval.

`search_documents` is the main tool. The model in Claude Desktop already writes
good answers, so the server only does retrieval; that makes answering free and
keeps the user's conversation context in one place.

## Setup (Windows, Claude Desktop)

1. Start the API: `docker compose up -d` and check `http://localhost:8000/health`.
2. Create a small virtual environment for the MCP server (Python 3.10+):

   ```powershell
   cd D:\Project\rag-pdf-chatbot
   python -m venv .venv-mcp
   .venv-mcp\Scripts\pip install -r mcp_server\requirements.txt
   ```

3. In Claude Desktop open **Settings > Developer > Edit Config** and add the
   `rag-pdf` entry from [`claude_desktop_config.example.json`](claude_desktop_config.example.json)
   (adjust the paths if your repo lives elsewhere).
4. Quit Claude Desktop fully (also from the system tray) and open it again. The
   tools appear under the tools/connectors menu in a new chat.
5. Try: *"What documents do I have?"* or *"Search my PDFs: how many days of annual leave?"*

macOS/Linux: same steps with `.venv-mcp/bin/python` and forward slashes.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `RAG_API_URL` | `http://localhost:8000` | where the FastAPI app runs (match `API_PORT`) |
| `RAG_API_TIMEOUT` | `120` | seconds per API call |

## Test and debug

```bash
pip install -r mcp_server/requirements.txt pytest
python -m pytest -q mcp_server/tests          # fake HTTP API: no Docker, no key needed
```

To poke at the tools by hand, run the MCP Inspector (needs Node.js):

```bash
npx @modelcontextprotocol/inspector .venv-mcp/Scripts/python.exe mcp_server/server.py
```

Claude Desktop writes server logs to `%APPDATA%\Claude\logs\mcp-server-rag-pdf.log`.
The server talks MCP over stdout, so it must never `print()`; logs go to stderr.

## Next steps

- Remote connector: run with `mcp.run("streamable-http")` behind HTTPS and add
  OAuth 2.1 so it can be added as a custom connector in claude.ai.
- A `resources` entry per document so the client can open a whole PDF's text.
