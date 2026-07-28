"""Search the FAISS index and return official evidence."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import (
    SentenceTransformer,
)

from src.config import (
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    load_settings,
)


class TeaRetriever:
    """Semantic retriever for official tea documents."""

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        chunks_path: Path = CHUNKS_PATH,
    ) -> None:
        if (
            not index_path.exists()
            or not chunks_path.exists()
        ):
            raise RuntimeError(
                "The RAG index is missing. Run:\n"
                "python -m src.rag.build_index"
            )

        self.settings = load_settings()

        self.index = faiss.read_index(
            str(index_path)
        )

        self.chunks = json.loads(
            chunks_path.read_text(
                encoding="utf-8"
            )
        )

        if self.index.ntotal != len(
            self.chunks
        ):
            raise RuntimeError(
                "The FAISS index and tea_chunks.json "
                "do not match. Rebuild the RAG index."
            )

        self.embedding_model = (
            SentenceTransformer(
                self.settings.embedding_model
            )
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        minimum_similarity: float | None = None,
    ) -> list[dict]:
        """Return the most relevant document chunks."""
        query = query.strip()

        if not query:
            return []

        number_of_results = min(
            top_k or self.settings.top_k,
            len(self.chunks),
        )

        similarity_limit = (
            self.settings.minimum_similarity
            if minimum_similarity is None
            else minimum_similarity
        )

        query_embedding = (
            self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
        )

        scores, indexes = self.index.search(
            np.ascontiguousarray(
                query_embedding
            ),
            number_of_results,
        )

        results: list[dict] = []

        for rank, (
            score,
            index_position,
        ) in enumerate(
            zip(
                scores[0],
                indexes[0],
            ),
            start=1,
        ):
            if index_position < 0:
                continue

            score_value = float(score)

            if score_value < similarity_limit:
                continue

            item = dict(
                self.chunks[
                    int(index_position)
                ]
            )

            item.update(
                {
                    "rank": rank,
                    "similarity_score": round(
                        score_value,
                        4,
                    ),
                    "source_id": f"S{rank}",
                }
            )

            results.append(item)

        return results