"""Groq language-model client used by the agents."""

from __future__ import annotations

import json
import re
from typing import Any

from groq import Groq

from src.config import get_groq_api_key


class LLMService:
    """Small wrapper around the Groq API."""

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        self.client = Groq(
            api_key=(
                api_key
                or get_groq_api_key()
            )
        )

    def complete_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Return a normal text response."""

        response = (
            self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=temperature,
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "The language model returned "
                "an empty response."
            )

        return content.strip()

    def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Request and return one JSON object."""

        response = (
            self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            system_prompt
                            + "\nReturn exactly one valid "
                            + "JSON object. Do not use "
                            + "Markdown code blocks."
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={
                    "type": "json_object"
                },
                temperature=temperature,
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "The language model returned "
                "an empty JSON response."
            )

        return self.parse_json(
            content
        )

    @staticmethod
    def parse_json(
        content: str,
    ) -> dict[str, Any]:
        """Convert the model response into a dictionary."""

        cleaned = content.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        try:
            result = json.loads(
                cleaned
            )

        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start < 0 or end <= start:
                raise RuntimeError(
                    "The model did not return "
                    "valid JSON."
                )

            result = json.loads(
                cleaned[start : end + 1]
            )

        if not isinstance(
            result,
            dict,
        ):
            raise RuntimeError(
                "The model response was not "
                "a JSON object."
            )

        return result