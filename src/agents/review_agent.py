"""Review Agent implementing the reflection pattern."""

from __future__ import annotations

import json
from typing import Any

from src.config import Settings
from src.llm_client import LLMService
from src.schemas import AgentMessage, Route


class ReviewAgent:
    """Check grounding before showing the answer."""

    name = "ReviewAgent"

    def __init__(
        self,
        llm: LLMService,
        settings: Settings,
    ) -> None:
        self.llm = llm
        self.settings = settings

    def handle(
        self,
        message: AgentMessage,
    ) -> AgentMessage:
        """Review data and document results."""
        question = str(
            message.payload.get(
                "question",
                "",
            )
        )

        route = Route(
            message.payload.get("route")
        )

        data_payload = message.payload.get(
            "data_payload"
        )

        document_payload = (
            message.payload.get(
                "document_payload"
            )
        )

        if route == Route.UNRELATED:
            return self.create_message(
                approved=True,
                final_answer=(
                    "This question is outside the "
                    "project scope. This assistant "
                    "supports Sri Lankan tea production, "
                    "exports, sales information, and "
                    "retrieval from the official "
                    "documents included in this project. "
                    "It does not provide predictions, "
                    "live auction prices, weather "
                    "forecasts, disease diagnosis, or "
                    "fertiliser advice."
                ),
                issues=[],
                sources=[],
            )

        deterministic_issues = (
            self.find_deterministic_issues(
                route,
                data_payload,
                document_payload,
            )
        )

        if deterministic_issues:
            return self.create_message(
                approved=False,
                final_answer=(
                    "A grounded answer could not be "
                    "created because: "
                    + "; ".join(
                        deterministic_issues
                    )
                ),
                issues=deterministic_issues,
                sources=self.collect_sources(
                    data_payload,
                    document_payload,
                ),
            )

        review_input = {
            "question": question,
            "route": route.value,
            "data_result": (
                (data_payload or {})
                .get("result")
            ),
            "document_draft": (
                (document_payload or {})
                .get("draft_answer")
            ),
            "retrieved_evidence": (
                (document_payload or {})
                .get("evidence", [])
            ),
        }

        system_prompt = """
You are the final Review Agent in a university Agentic AI project.

Check the proposed information before it is shown to the user.

Rules:
1. Every numerical claim must be supported by data_result.
2. Every document-based statement must be supported by retrieved_evidence.
3. Do not use outside knowledge.
4. Do not invent values, years, causes, page numbers, units, or sources.
5. For a combined route, combine the deterministic data summary with
   the official-document explanation.
6. When retrieved evidence does not explain a numerical change, clearly
   say that the evidence does not provide an explanation.
7. Use simple, professional English.

Return JSON:
{
  "approved": true,
  "final_answer": "fully grounded final answer",
  "issues": [],
  "used_source_ids": ["S1"]
}
""".strip()

        try:
            decision = self.llm.complete_json(
                model=(
                    self.settings.reasoning_model
                ),
                system_prompt=system_prompt,
                user_prompt=json.dumps(
                    review_input,
                    ensure_ascii=False,
                ),
                temperature=0.0,
            )

            approved = bool(
                decision.get(
                    "approved",
                    False,
                )
            )

            final_answer = str(
                decision.get(
                    "final_answer",
                    "",
                )
            ).strip()

            issues = decision.get(
                "issues",
                [],
            )

            if not isinstance(issues, list):
                issues = [str(issues)]

            if not final_answer:
                approved = False

                issues.append(
                    "The Review Agent returned "
                    "an empty final answer."
                )

                final_answer = (
                    self.fallback_answer(
                        data_payload,
                        document_payload,
                    )
                )

        except Exception as error:
            final_answer = self.fallback_answer(
                data_payload,
                document_payload,
            )

            if route in {
                Route.ANNUAL_PRODUCTION,
                Route.MONTHLY_PRODUCTION,
                Route.EXPORT_ANALYSIS,
            }:
                approved = True

            else:
                approved = False

            issues = [
                "The language-model review could not "
                f"be completed: {error}"
            ]

        return self.create_message(
            approved=approved,
            final_answer=final_answer,
            issues=issues,
            sources=self.collect_sources(
                data_payload,
                document_payload,
            ),
        )

    @staticmethod
    def find_deterministic_issues(
        route: Route,
        data_payload: dict[str, Any] | None,
        document_payload: dict[str, Any] | None,
    ) -> list[str]:
        """Check whether required worker results exist."""
        issues: list[str] = []

        requires_data = route in {
            Route.ANNUAL_PRODUCTION,
            Route.MONTHLY_PRODUCTION,
            Route.EXPORT_ANALYSIS,
            Route.COMBINED_ANALYSIS,
        }

        requires_documents = route in {
            Route.DOCUMENT_SEARCH,
            Route.COMBINED_ANALYSIS,
        }

        if requires_data:
            if (
                not data_payload
                or data_payload.get("status")
                != "success"
            ):
                issues.append(
                    (data_payload or {}).get(
                        "error",
                        "The Data Agent did not "
                        "return a successful result.",
                    )
                )

        if requires_documents:
            if not document_payload:
                issues.append(
                    "The Document Agent did not "
                    "return a result."
                )

            elif (
                document_payload.get("status")
                != "success"
            ):
                issues.append(
                    document_payload.get(
                        "error",
                        "No official-document evidence "
                        "was retrieved.",
                    )
                )

            elif not document_payload.get(
                "evidence"
            ):
                issues.append(
                    "No official-document evidence "
                    "was retrieved."
                )

        return issues

    @staticmethod
    def fallback_answer(
        data_payload: dict[str, Any] | None,
        document_payload: dict[str, Any] | None,
    ) -> str:
        """Create a conservative answer without adding new facts."""
        answer_parts: list[str] = []

        if (
            data_payload
            and data_payload.get("result")
        ):
            summary = data_payload[
                "result"
            ].get("summary")

            if summary:
                answer_parts.append(str(summary))

        if document_payload:
            document_answer = (
                document_payload.get(
                    "draft_answer"
                )
            )

            if document_answer:
                answer_parts.append(
                    str(document_answer)
                )

        if not answer_parts:
            return (
                "No grounded answer is currently "
                "available."
            )

        return "\n\n".join(answer_parts)

    @staticmethod
    def collect_sources(
        data_payload: dict[str, Any] | None,
        document_payload: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Collect dataset and document sources."""
        sources: list[dict[str, Any]] = []

        if (
            data_payload
            and data_payload.get("result")
        ):
            for source in data_payload[
                "result"
            ].get("sources", []):
                sources.append(
                    {
                        "type": "dataset",
                        "source": source,
                    }
                )

        if document_payload:
            for item in document_payload.get(
                "evidence",
                [],
            ):
                sources.append(
                    {
                        "type": "document",
                        "source_id": item.get(
                            "source_id"
                        ),
                        "document": item.get(
                            "document"
                        ),
                        "page": item.get(
                            "page"
                        ),
                        "similarity_score": item.get(
                            "similarity_score"
                        ),
                    }
                )

        unique_sources: list[
            dict[str, Any]
        ] = []

        seen: set[str] = set()

        for source in sources:
            source_key = json.dumps(
                source,
                sort_keys=True,
            )

            if source_key not in seen:
                seen.add(source_key)
                unique_sources.append(source)

        return unique_sources

    def create_message(
        self,
        *,
        approved: bool,
        final_answer: str,
        issues: list[str],
        sources: list[dict[str, Any]],
    ) -> AgentMessage:
        """Create the final review decision message."""
        return AgentMessage(
            sender=self.name,
            receiver="Orchestrator",
            message_type="REVIEW_DECISION",
            payload={
                "approved": approved,
                "final_answer": final_answer,
                "issues": issues,
                "sources": sources,
                "model": (
                    self.settings.reasoning_model
                ),
            },
        )