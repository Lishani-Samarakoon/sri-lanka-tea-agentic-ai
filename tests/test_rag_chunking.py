import pytest

from src.rag.build_index import (
    clean_text,
    split_text_into_chunks,
)


def test_clean_text_removes_extra_spaces():
    result = clean_text(
        "Tea    production\n\n\nreport"
    )

    assert result == (
        "Tea production\n\nreport"
    )


def test_chunking_creates_multiple_chunks():
    text = " ".join(
        ["tea"] * 300
    )

    chunks = split_text_into_chunks(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1

    assert all(
        chunk
        for chunk in chunks
    )


def test_invalid_overlap_is_rejected():
    with pytest.raises(
        ValueError
    ):
        split_text_into_chunks(
            "sample text",
            chunk_size=10,
            overlap=10,
        )