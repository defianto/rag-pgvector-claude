"""Ask Claude to answer a question using only the retrieved chunks."""

import anthropic

from app import config
from app.db import SearchHit

SYSTEM_PROMPT = """You answer questions about the user's uploaded documents.

Use only the numbered sources given in the user message. After each claim,
cite the source it came from, like [1] or [2][3]. If the sources do not
contain the answer, say so plainly instead of guessing. Answer in the same
language as the question, and keep it short and direct."""

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY from the environment.
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    sources = "\n\n".join(
        f'<source id="{i}" file="{h.filename}" page="{h.page}">\n{h.content}\n</source>'
        for i, h in enumerate(hits, start=1)
    )
    return f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}"


def answer(question: str, hits: list[SearchHit]) -> str:
    if not hits:
        return "No documents have been uploaded yet, so there is nothing to search."

    response = _get_client().beta.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=config.CLAUDE_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
        output_config={"effort": config.CLAUDE_EFFORT},
        # If the model declines on a safety false positive, the API retries
        # on a fallback model inside the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        return "Sorry, I can't answer that question."

    return "".join(b.text for b in response.content if b.type == "text").strip()
