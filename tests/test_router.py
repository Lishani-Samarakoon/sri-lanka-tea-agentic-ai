"""Tests for the Router Agent."""

from src.agents.router_agent import (
    RouterAgent,
)
from src.config import Settings
from src.schemas import AgentMessage


class FakeLLM:
    """Fake language model for testing."""

    def complete_json(
        self,
        **kwargs,
    ):
        return {
            "route": "ANNUAL_PRODUCTION",
            "confidence": 0.98,
            "reason": (
                "The question requests "
                "annual production analysis."
            ),
            "required_agents": [
                "DATA_AGENT",
                "REVIEW_AGENT",
            ],
        }


def test_router_returns_annual_route() -> None:
    """The Router should return a structured route."""

    agent = RouterAgent(
        llm=FakeLLM(),
        settings=Settings(),
    )

    request = AgentMessage(
        sender="Orchestrator",
        receiver="RouterAgent",
        message_type="ROUTE_REQUEST",
        payload={
            "question": (
                "Which year had the highest "
                "annual tea production?"
            )
        },
    )

    response = agent.handle(
        request
    )

    assert (
        response.message_type
        == "ROUTE_DECISION"
    )

    assert (
        response.payload["route"]
        == "ANNUAL_PRODUCTION"
    )

    assert (
        response.payload["confidence"]
        == 0.98
    )

    assert (
        response.sender
        == "RouterAgent"
    )