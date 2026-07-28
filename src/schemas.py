"""Shared data structures used for agent-to-agent communication."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class Route(str, Enum):
    """Routes supported by the Router Agent."""

    ANNUAL_PRODUCTION = "ANNUAL_PRODUCTION"
    MONTHLY_PRODUCTION = "MONTHLY_PRODUCTION"
    EXPORT_ANALYSIS = "EXPORT_ANALYSIS"
    DOCUMENT_SEARCH = "DOCUMENT_SEARCH"
    COMBINED_ANALYSIS = "COMBINED_ANALYSIS"
    UNRELATED = "UNRELATED"


@dataclass
class AgentMessage:
    """
    Structured message exchanged between agents.

    This is the project's custom agent communication protocol.
    """

    sender: str
    receiver: str
    message_type: str
    payload: dict[str, Any]

    message_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the message into a normal dictionary."""
        return asdict(self)


def json_safe(value: Any) -> Any:
    """
    Convert pandas and NumPy values into normal Python values.

    This prevents errors when Streamlit displays JSON.
    """
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    return value