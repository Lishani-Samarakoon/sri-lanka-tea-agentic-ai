"""Orchestrator coordinating all specialist agents."""

from __future__ import annotations

from src.agents.data_agent import DataAgent
from src.agents.document_agent import (
    DocumentAgent,
)
from src.agents.review_agent import ReviewAgent
from src.agents.router_agent import RouterAgent
from src.config import Settings, load_settings
from src.llm_client import LLMService
from src.schemas import AgentMessage, Route


class TeaOrchestrator:
    """Implement the orchestrator-worker pattern."""

    name = "Orchestrator"

    def __init__(
        self,
        llm: LLMService | None = None,
        settings: Settings | None = None,
        router_agent: RouterAgent | None = None,
        data_agent: DataAgent | None = None,
        document_agent: (
            DocumentAgent | None
        ) = None,
        review_agent: ReviewAgent | None = None,
    ) -> None:
        self.settings = (
            settings
            or load_settings()
        )

        self.llm = (
            llm
            or LLMService()
        )

        self.router_agent = (
            router_agent
            or RouterAgent(
                self.llm,
                self.settings,
            )
        )

        self.data_agent = (
            data_agent
            or DataAgent(
                self.settings
            )
        )

        self.document_agent = (
            document_agent
            or DocumentAgent(
                self.llm,
                self.settings,
            )
        )

        self.review_agent = (
            review_agent
            or ReviewAgent(
                self.llm,
                self.settings,
            )
        )

    def run(
        self,
        question: str,
    ) -> dict:
        """Run the complete agent communication workflow."""
        question = question.strip()

        if not question:
            raise ValueError(
                "Please enter a question."
            )

        trace: list[dict] = []

        user_message = AgentMessage(
            sender="User",
            receiver=self.name,
            message_type="USER_QUESTION",
            payload={
                "question": question,
            },
        )

        trace.append(
            user_message.to_dict()
        )

        router_request = AgentMessage(
            sender=self.name,
            receiver="RouterAgent",
            message_type="ROUTE_REQUEST",
            payload={
                "question": question,
            },
        )

        trace.append(
            router_request.to_dict()
        )

        router_response = (
            self.router_agent.handle(
                router_request
            )
        )

        trace.append(
            router_response.to_dict()
        )

        selected_route = Route(
            router_response.payload["route"]
        )

        data_response = None
        document_response = None

        data_routes = {
            Route.ANNUAL_PRODUCTION,
            Route.MONTHLY_PRODUCTION,
            Route.EXPORT_ANALYSIS,
            Route.COMBINED_ANALYSIS,
        }

        document_routes = {
            Route.DOCUMENT_SEARCH,
            Route.COMBINED_ANALYSIS,
        }

        if selected_route in data_routes:
            data_request = AgentMessage(
                sender=self.name,
                receiver="DataAgent",
                message_type="DATA_REQUEST",
                payload={
                    "question": question,
                    "route": (
                        selected_route.value
                    ),
                },
            )

            trace.append(
                data_request.to_dict()
            )

            data_response = (
                self.data_agent.handle(
                    data_request
                )
            )

            trace.append(
                data_response.to_dict()
            )

        if selected_route in document_routes:
            document_request = AgentMessage(
                sender=self.name,
                receiver="DocumentAgent",
                message_type=(
                    "DOCUMENT_REQUEST"
                ),
                payload={
                    "question": question,
                    "route": (
                        selected_route.value
                    ),
                },
            )

            trace.append(
                document_request.to_dict()
            )

            document_response = (
                self.document_agent.handle(
                    document_request
                )
            )

            trace.append(
                document_response.to_dict()
            )

        review_request = AgentMessage(
            sender=self.name,
            receiver="ReviewAgent",
            message_type="REVIEW_REQUEST",
            payload={
                "question": question,
                "route": selected_route.value,
                "data_payload": (
                    data_response.payload
                    if data_response
                    else None
                ),
                "document_payload": (
                    document_response.payload
                    if document_response
                    else None
                ),
            },
        )

        trace.append(
            review_request.to_dict()
        )

        review_response = (
            self.review_agent.handle(
                review_request
            )
        )

        trace.append(
            review_response.to_dict()
        )

        return {
            "question": question,
            "route": selected_route.value,
            "router": (
                router_response.payload
            ),
            "approved": (
                review_response.payload[
                    "approved"
                ]
            ),
            "answer": (
                review_response.payload[
                    "final_answer"
                ]
            ),
            "issues": (
                review_response.payload[
                    "issues"
                ]
            ),
            "sources": (
                review_response.payload[
                    "sources"
                ]
            ),
            "data_result": (
                data_response.payload.get(
                    "result"
                )
                if (
                    data_response
                    and data_response.payload.get(
                        "status"
                    ) == "success"
                )
                else None
            ),
            "evidence": (
                document_response.payload.get(
                    "evidence",
                    [],
                )
                if document_response
                else []
            ),
            "trace": trace,
            "models": {
                "routing": (
                    self.settings.router_model
                ),
                "document_answering": (
                    self.settings.reasoning_model
                ),
                "review": (
                    self.settings.reasoning_model
                ),
                "embeddings": (
                    self.settings.embedding_model
                ),
                "data_calculation": (
                    "Deterministic pandas tools"
                ),
            },
        }