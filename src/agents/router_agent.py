"""Router Agent for classifying tea-industry questions."""

from __future__ import annotations

import re

from src.config import Settings
from src.llm_client import LLMService
from src.schemas import AgentMessage, Route


class RouterAgent:
    """Select the correct worker agent for each question."""

    name = "RouterAgent"

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
        """Process a routing request."""
        question = str(
            message.payload.get(
                "question",
                "",
            )
        ).strip()

        system_prompt = """
You are the Router Agent for a Sri Lankan tea-industry assistant.

Select exactly one route:

ANNUAL_PRODUCTION:
Questions requiring numerical annual tea-production data.

MONTHLY_PRODUCTION:
Questions requiring numerical monthly production or high-grown,
medium-grown, or low-grown production data.

EXPORT_ANALYSIS:
Questions requiring numerical export volume or export revenue data.

DOCUMENT_SEARCH:
Questions answered from official Sri Lanka Tea Board reports,
publications, policies, sales information, or industry documents.

COMBINED_ANALYSIS:
Questions requiring both a numerical dataset calculation and an
explanation retrieved from official documents.

UNRELATED:
Anything outside Sri Lankan tea production, exports, sales, or the
official documents included in this project.

Always use UNRELATED for:
- predictions
- production forecasting
- live auction prices
- weather forecasting
- disease diagnosis
- fertiliser or fertilizer advice
- personal financial or investment advice
- questions unrelated to the Sri Lankan tea industry

Return JSON with:
{
  "route": "ROUTE_NAME",
  "confidence": 0.0,
  "reason": "brief reason",
  "required_agents": ["AGENT_NAME"]
}
""".strip()

        try:
            decision = self.llm.complete_json(
                model=self.settings.router_model,
                system_prompt=system_prompt,
                user_prompt=(
                    f"User question: {question}"
                ),
            )

            route = Route(
                str(
                    decision.get(
                        "route",
                        "UNRELATED",
                    )
                ).upper()
            )

            confidence = float(
                decision.get(
                    "confidence",
                    0.0,
                )
            )

            reason = str(
                decision.get(
                    "reason",
                    "No reason supplied.",
                )
            )

            required_agents = decision.get(
                "required_agents",
                [],
            )

            if not isinstance(
                required_agents,
                list,
            ):
                required_agents = []

        except Exception as error:
            route = self.fallback_route(
                question
            )

            confidence = 0.55

            reason = (
                "The deterministic fallback router "
                f"was used because: {error}"
            )

            required_agents = []

        if not required_agents:
            required_agents = (
                self.required_agents(route)
            )

        return AgentMessage(
            sender=self.name,
            receiver="Orchestrator",
            message_type="ROUTE_DECISION",
            payload={
                "question": question,
                "route": route.value,
                "confidence": max(
                    0.0,
                    min(1.0, confidence),
                ),
                "reason": reason,
                "required_agents": (
                    required_agents
                ),
                "model": (
                    self.settings.router_model
                ),
            },
        )

    @staticmethod
    def required_agents(
        route: Route,
    ) -> list[str]:
        """Return the agents needed for a route."""
        if route == Route.COMBINED_ANALYSIS:
            return [
                "DATA_AGENT",
                "DOCUMENT_AGENT",
                "REVIEW_AGENT",
            ]

        if route in {
            Route.ANNUAL_PRODUCTION,
            Route.MONTHLY_PRODUCTION,
            Route.EXPORT_ANALYSIS,
        }:
            return [
                "DATA_AGENT",
                "REVIEW_AGENT",
            ]

        if route == Route.DOCUMENT_SEARCH:
            return [
                "DOCUMENT_AGENT",
                "REVIEW_AGENT",
            ]

        return ["REVIEW_AGENT"]

    @staticmethod
    def fallback_route(
        question: str,
    ) -> Route:
        """Use keywords when the router model is unavailable."""
        text = question.lower()

        excluded_terms = [
            "predict",
            "prediction",
            "forecast",
            "future production",
            "auction price",
            "live price",
            "weather",
            "disease",
            "fertiliser",
            "fertilizer",
            "investment advice",
        ]

        if any(
            term in text
            for term in excluded_terms
        ):
            return Route.UNRELATED

        document_terms = [
            "report",
            "document",
            "according to",
            "official",
            "explain",
            "reason",
            "policy",
            "publication",
            "what does the report",
        ]

        has_document_request = any(
            term in text
            for term in document_terms
        )

        has_year = bool(
            re.search(
                r"\b(?:19|20)\d{2}\b",
                text,
            )
        )

        numeric_terms = [
            "highest",
            "lowest",
            "compare",
            "difference",
            "change",
            "percentage",
            "how much",
        ]

        has_numeric_request = (
            has_year
            or any(
                term in text
                for term in numeric_terms
            )
        )

        if (
            has_document_request
            and has_numeric_request
        ):
            return Route.COMBINED_ANALYSIS

        if any(
            term in text
            for term in [
                "export",
                "exports",
                "revenue",
                "earnings",
            ]
        ):
            return Route.EXPORT_ANALYSIS

        month_names = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]

        if (
            any(
                month in text
                for month in month_names
            )
            or "monthly" in text
            or "high grown" in text
            or "medium grown" in text
            or "low grown" in text
            or "elevation" in text
        ):
            return Route.MONTHLY_PRODUCTION

        if (
            "production" in text
            and (
                "year" in text
                or has_year
                or any(
                    term in text
                    for term in numeric_terms
                )
            )
        ):
            return Route.ANNUAL_PRODUCTION

        if has_document_request:
            return Route.DOCUMENT_SEARCH

        return Route.UNRELATED