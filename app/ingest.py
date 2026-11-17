"""PDF -> text -> overlapping chunks."""

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass
class Chunk:
    page: int  # 1-based page number
    index: int  # position of the chunk within the document
    content: str


def extract_pages(pdf_bytes: bytes) -> list[str]:
    """Return the text of each page (empty string for pages without text)."""
    reader = PdfReader(BytesIO(pdf_bytes))
    return [(page.extract_text() or "") for page in reader.pages]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into pieces of about `size` characters.

    Neighbouring pieces share `overlap` characters so a sentence that falls
    on a boundary is still found whole in at least one piece. Cuts are moved
    back to the nearest whitespace so words are not broken in half.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = " ".join(text.split())  # normalise whitespace
    if not text:
        return []

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        # start the next piece at a word boundary too
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return [p for p in pieces if p]


def chunk_pdf(pdf_bytes: bytes, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, page_text in enumerate(extract_pages(pdf_bytes), start=1):
        for piece in split_text(page_text, size, overlap):
            chunks.append(Chunk(page=page_no, index=len(chunks), content=piece))
    return chunks
