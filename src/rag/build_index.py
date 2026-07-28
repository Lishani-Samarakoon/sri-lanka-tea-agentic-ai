"""Create a FAISS index from the official tea-industry PDFs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.config import (
    BASE_DIR,
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    INDEX_METADATA_PATH,
    RAG_SOURCE_DIRS,
    VECTOR_STORE_DIR,
    load_settings,
)


def find_pdf_files(
    roots: Iterable[Path] | Path | None = None,
) -> list[Path]:
    """
    Find PDFs in both existing report locations.

    The function searches:
    1. The root annual_reports folder.
    2. The documents folder and all its subfolders.
    """
    if roots is None:
        selected_roots = list(RAG_SOURCE_DIRS)

    elif isinstance(roots, Path):
        selected_roots = [roots]

    else:
        selected_roots = list(roots)

    found: dict[str, Path] = {}

    for root in selected_roots:
        if not root.exists():
            continue

        for pdf_path in root.rglob("*.pdf"):
            if pdf_path.is_file():
                found[
                    str(pdf_path.resolve())
                ] = pdf_path

    return sorted(
        found.values(),
        key=lambda path: str(path).lower(),
    )


def clean_text(text: str) -> str:
    """Remove unnecessary spaces from PDF text."""
    text = text.replace("\x00", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def extract_pdf_pages(
    pdf_path: Path,
) -> list[dict]:
    """Extract text from each page of one PDF."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))

    pages: list[dict] = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        page_text = clean_text(
            page.extract_text() or ""
        )

        if page_text:
            pages.append(
                {
                    "page": page_number,
                    "text": page_text,
                }
            )

    return pages


def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """Split text into overlapping sections."""
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be zero or greater "
            "and smaller than chunk_size."
        )

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(cleaned_text):
        expected_end = min(
            start + chunk_size,
            len(cleaned_text),
        )

        end = expected_end

        if expected_end < len(cleaned_text):
            search_start = (
                start
                + int(chunk_size * 0.60)
            )

            possible_boundaries = [
                cleaned_text.rfind(
                    "\n",
                    search_start,
                    expected_end,
                ),
                cleaned_text.rfind(
                    ". ",
                    search_start,
                    expected_end,
                ),
                cleaned_text.rfind(
                    " ",
                    search_start,
                    expected_end,
                ),
            ]

            boundary = max(
                possible_boundaries
            )

            if boundary > start:
                end = boundary + 1

        chunk = cleaned_text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned_text):
            break

        next_start = end - overlap

        start = max(
            start + 1,
            next_start,
        )

    return chunks


def infer_category(pdf_path: Path) -> str:
    """Infer a document category from its folder name."""
    path_parts = {
        part.lower()
        for part in pdf_path.parts
    }

    if "annual_reports" in path_parts:
        return "annual_report"

    if "production_reports" in path_parts:
        return "production_report"

    if "export_reports" in path_parts:
        return "export_report"

    return "official_document"


def infer_year(pdf_path: Path) -> int | None:
    """Find a four-digit year in the PDF file name."""
    match = re.search(
        r"(?:19|20)\d{2}",
        pdf_path.name,
    )

    if not match:
        return None

    return int(match.group())


def safe_relative_path(
    pdf_path: Path,
) -> str:
    """Create a project-relative path for metadata."""
    try:
        return str(
            pdf_path.relative_to(BASE_DIR)
        )

    except ValueError:
        return str(pdf_path)


def create_chunks(
    pdf_files: Iterable[Path],
) -> list[dict]:
    """Convert PDF pages into source-labelled chunks."""
    chunks: list[dict] = []

    for pdf_path in pdf_files:
        print(
            f"Reading: {safe_relative_path(pdf_path)}"
        )

        pages = extract_pdf_pages(pdf_path)

        for page_information in pages:
            page_chunks = split_text_into_chunks(
                page_information["text"]
            )

            for chunk_number, text in enumerate(
                page_chunks,
                start=1,
            ):
                chunks.append(
                    {
                        "chunk_id": (
                            f"C{len(chunks) + 1:06d}"
                        ),
                        "text": text,
                        "document": pdf_path.name,
                        "relative_path": (
                            safe_relative_path(
                                pdf_path
                            )
                        ),
                        "page": page_information[
                            "page"
                        ],
                        "chunk_number_on_page": (
                            chunk_number
                        ),
                        "category": infer_category(
                            pdf_path
                        ),
                        "year": infer_year(
                            pdf_path
                        ),
                    }
                )

    return chunks


def build_index() -> dict:
    """Build and save the FAISS vector database."""
    import faiss
    import numpy as np
    from sentence_transformers import (
        SentenceTransformer,
    )

    settings = load_settings()

    pdf_files = find_pdf_files()

    if not pdf_files:
        raise RuntimeError(
            "No PDF files were found. Add official "
            "PDFs to annual_reports or documents."
        )

    chunks = create_chunks(pdf_files)

    if not chunks:
        raise RuntimeError(
            "No extractable PDF text was found. "
            "Image-only scanned PDFs require OCR "
            "before this program can index them."
        )

    embedding_model = SentenceTransformer(
        settings.embedding_model
    )

    text_values = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        text_values,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        np.ascontiguousarray(
            embeddings
        )
    )

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH),
    )

    CHUNKS_PATH.write_text(
        json.dumps(
            chunks,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "built_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "embedding_model": (
            settings.embedding_model
        ),
        "document_count": len(pdf_files),
        "chunk_count": len(chunks),
        "vector_dimension": int(
            embeddings.shape[1]
        ),
        "index_type": (
            "FAISS IndexFlatIP with "
            "normalised embeddings"
        ),
        "searched_folders": [
            str(path.relative_to(BASE_DIR))
            if path.is_relative_to(BASE_DIR)
            else str(path)
            for path in RAG_SOURCE_DIRS
        ],
    }

    INDEX_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata


def main() -> None:
    """Run index building from the terminal."""
    metadata = build_index()

    print("\nRAG index built successfully.")
    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()