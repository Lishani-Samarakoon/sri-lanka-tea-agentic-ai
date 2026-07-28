"""Run five retrieval evaluations and save the results."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.rag.retriever import TeaRetriever


QUERIES_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "rag_evaluation_queries.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "rag_evaluation_results.csv"
)


def normalise_name(
    text: str,
) -> str:
    """Prepare a document name for comparison."""
    return (
        text.lower()
        .replace(".pdf", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def main() -> None:
    """Evaluate top-three document retrieval."""
    queries = json.loads(
        QUERIES_PATH.read_text(
            encoding="utf-8"
        )
    )

    retriever = TeaRetriever()

    rows: list[dict] = []

    for item in queries:
        results = retriever.retrieve(
            item["query"],
            top_k=3,
            minimum_similarity=-1.0,
        )

        expected_document = (
            normalise_name(
                item["expected_document"]
            )
        )

        result_names = [
            normalise_name(
                result["document"]
            )
            for result in results
        ]

        top_1_document = (
            results[0]["document"]
            if results
            else ""
        )

        top_1_match = (
            bool(result_names)
            and (
                expected_document
                in result_names[0]
                or result_names[0]
                in expected_document
            )
        )

        top_3_match = any(
            expected_document in name
            or name in expected_document
            for name in result_names
        )

        combined_text = " ".join(
            result["text"]
            for result in results
        ).lower()

        expected_keywords = (
            item.get(
                "expected_keywords",
                [],
            )
        )

        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower()
            in combined_text
        ]

        keyword_coverage = (
            len(matched_keywords)
            / len(expected_keywords)
            if expected_keywords
            else 0
        )

        rows.append(
            {
                "query_id": item["id"],
                "query": item["query"],
                "expected_document": (
                    item["expected_document"]
                ),
                "top_1_document": (
                    top_1_document
                ),
                "top_1_match": (
                    "yes"
                    if top_1_match
                    else "no"
                ),
                "top_3_match": (
                    "yes"
                    if top_3_match
                    else "no"
                ),
                "matched_keywords": (
                    ", ".join(
                        matched_keywords
                    )
                ),
                "keyword_coverage": round(
                    keyword_coverage,
                    2,
                ),
                "manual_answer_supported": (
                    "REVIEW_REQUIRED"
                ),
                "notes": "",
            }
        )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        fieldnames = list(
            rows[0].keys()
        )

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "Evaluation completed."
    )

    print(
        f"Results saved to: {RESULTS_PATH}"
    )

    print(
        "Open the CSV and manually replace "
        "REVIEW_REQUIRED with yes or no."
    )


if __name__ == "__main__":
    main()