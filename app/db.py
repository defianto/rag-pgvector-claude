"""PostgreSQL + pgvector access."""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from app import config
from app.ingest import Chunk


@dataclass
class SearchHit:
    filename: str
    page: int
    content: str
    score: float  # cosine similarity, 1.0 = identical


@contextmanager
def connect():
    with psycopg.connect(config.DATABASE_URL) as conn:
        register_vector(conn)
        yield conn


def save_document(filename: str, chunks: list[Chunk], vectors: list[np.ndarray]) -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (filename) VALUES (%s) RETURNING id", (filename,)
        )
        doc_id = cur.fetchone()[0]
        cur.executemany(
            """
            INSERT INTO chunks (document_id, page, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (doc_id, c.page, c.index, c.content, v)
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )
        return doc_id


def list_documents() -> list[dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d.filename, d.uploaded_at, count(c.id)
            FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id ORDER BY d.uploaded_at DESC
            """
        )
        return [
            {"id": r[0], "filename": r[1], "uploaded_at": r[2], "chunks": r[3]}
            for r in cur.fetchall()
        ]


def delete_document(doc_id: int) -> bool:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        return cur.rowcount > 0


def search(query_vector: np.ndarray, top_k: int, document_id: int | None = None) -> list[SearchHit]:
    """Nearest chunks by cosine distance (`<=>`), optionally within one document."""
    sql = """
        SELECT d.filename, c.page, c.content, 1 - (c.embedding <=> %(q)s) AS score
        FROM chunks c JOIN documents d ON d.id = c.document_id
        {where}
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s
    """
    where = "WHERE c.document_id = %(doc)s" if document_id is not None else ""
    params = {"q": query_vector, "k": top_k, "doc": document_id}
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql.format(where=where), params)
        return [SearchHit(r[0], r[1], r[2], float(r[3])) for r in cur.fetchall()]
