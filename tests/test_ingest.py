import pytest

from app.db import SearchHit
from app.ingest import split_text
from app.llm import build_prompt


def test_empty_text_gives_no_chunks():
    assert split_text("   \n  ", 100, 20) == []


def test_short_text_is_one_chunk():
    assert split_text("hello   world", 100, 20) == ["hello world"]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, 200, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # neighbouring chunks share some text
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-1] in b
    # nothing is lost: last word is present
    assert chunks[-1].endswith("word499")


def test_words_are_not_cut():
    text = " ".join(["abcdefghij"] * 100)
    for c in split_text(text, 95, 20):
        assert all(w == "abcdefghij" for w in c.split())


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)


def test_prompt_numbers_sources():
    hits = [SearchHit("a.pdf", 2, "first", 0.9), SearchHit("b.pdf", 7, "second", 0.8)]
    prompt = build_prompt("What?", hits)
    assert '<source id="1" file="a.pdf" page="2">' in prompt
    assert '<source id="2" file="b.pdf" page="7">' in prompt
    assert prompt.endswith("Question: What?")
