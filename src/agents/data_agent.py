"""Data Analysis Agent using deterministic pandas tools."""

from __future__ import annotations

import re
from typing import Any

from src.config import Settings
from src.schemas import AgentMessage, Route
from src.tools.data_tools import (
    DataRepository,
    DataValidationError,
    MONTH_ORDER,
)


class DataAgent:
    """Plan and execute one data-analysis operation."""

    name = "DataAgent"

    def __init__(
        self,
        settings: Settings,
        repository: DataRepository | None = None,
    ) -> None:
        self.settings = settings

        self.repository = (
            repository
            or DataRepository()
        )

    def handle(
        self,
        message: AgentMessage,
    ) -> AgentMessage:
        """Handle a data request from the orchestrator."""
        question = str(
            message.payload.get(
                "question",
                "",
            )
        ).strip()

        route = Route(
            message.payload.get("route")
        )

        try:
            plan = self.create_plan(
                question,
                route,
            )

            result = self.execute(plan)

            payload = {
                "status": "success",
                "route": route.value,
                "plan": plan,
                "result": result,
                "tool": "pandas",
            }

        except (
            DataValidationError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
        ) as error:
            payload = {
                "status": "error",
                "route": route.value,
                "error": str(error),
                "tool": "pandas",
            }

        except Exception as error:
            payload = {
                "status": "error",
                "route": route.value,
                "error": (
                    "Unexpected Data Agent error: "
                    f"{error}"
                ),
                "tool": "pandas",
            }

        return AgentMessage(
            sender=self.name,
            receiver="Orchestrator",
            message_type="DATA_RESULT",
            payload=payload,
        )

    @staticmethod
    def create_plan(
        question: str,
        route: Route,
    ) -> dict[str, Any]:
        """Create a deterministic tool-use plan."""
        text = question.lower()

        years = [
            int(year)
            for year in re.findall(
                r"\b(?:19|20)\d{2}\b",
                text,
            )
        ]

        months = [
            month
            for month in MONTH_ORDER
            if month.lower() in text
        ]

        direction: str | None = None

        if any(
            word in text
            for word in [
                "highest",
                "maximum",
                "most",
                "largest",
            ]
        ):
            direction = "highest"

        elif any(
            word in text
            for word in [
                "lowest",
                "minimum",
                "least",
                "smallest",
            ]
        ):
            direction = "lowest"

        if route == Route.COMBINED_ANALYSIS:
            if any(
                word in text
                for word in [
                    "export",
                    "revenue",
                    "earning",
                ]
            ):
                selected_route = (
                    Route.EXPORT_ANALYSIS
                )

            elif (
                months
                or "monthly" in text
                or "elevation" in text
                or "high grown" in text
                or "medium grown" in text
                or "low grown" in text
            ):
                selected_route = (
                    Route.MONTHLY_PRODUCTION
                )

            else:
                selected_route = (
                    Route.ANNUAL_PRODUCTION
                )

        else:
            selected_route = route

        if (
            selected_route
            == Route.ANNUAL_PRODUCTION
        ):
            if len(years) >= 2:
                return {
                    "operation": "compare_annual",
                    "parameters": {
                        "year_1": years[0],
                        "year_2": years[1],
                    },
                    "reason": (
                        "Two annual years were found."
                    ),
                }

            if len(years) == 1:
                return {
                    "operation": "get_annual",
                    "parameters": {
                        "year": years[0],
                    },
                    "reason": (
                        "One annual year was found."
                    ),
                }

            if direction:
                return {
                    "operation": (
                        f"{direction}_annual"
                    ),
                    "parameters": {},
                    "reason": (
                        "An annual extreme was requested."
                    ),
                }

            return {
                "operation": "annual_trend",
                "parameters": {},
                "reason": (
                    "A general annual trend was requested."
                ),
            }

        if (
            selected_route
            == Route.MONTHLY_PRODUCTION
        ):
            metric = "total_kg"

            if (
                "high grown" in text
                or "high elevation" in text
            ):
                metric = "high_kg"

            elif (
                "medium grown" in text
                or "medium elevation" in text
            ):
                metric = "medium_kg"

            elif (
                "low grown" in text
                or "low elevation" in text
            ):
                metric = "low_kg"

            selected_year = (
                years[0]
                if years
                else None
            )

            if len(months) >= 2:
                return {
                    "operation": "compare_months",
                    "parameters": {
                        "year": selected_year,
                        "month_1": months[0],
                        "month_2": months[1],
                        "metric": metric,
                    },
                    "reason": (
                        "Two months were found."
                    ),
                }

            if len(months) == 1:
                return {
                    "operation": "get_month",
                    "parameters": {
                        "year": selected_year,
                        "month": months[0],
                        "metric": metric,
                    },
                    "reason": (
                        "One month was found."
                    ),
                }

            if direction:
                return {
                    "operation": (
                        f"{direction}_month"
                    ),
                    "parameters": {
                        "year": selected_year,
                        "metric": metric,
                    },
                    "reason": (
                        "A monthly extreme was requested."
                    ),
                }

            return {
                "operation": "monthly_trend",
                "parameters": {
                    "year": selected_year,
                },
                "reason": (
                    "A monthly trend was requested."
                ),
            }

        if (
            selected_route
            == Route.EXPORT_ANALYSIS
        ):
            metric = "export_volume_mn_kg"

            if any(
                word in text
                for word in [
                    "lkr",
                    "rupee",
                    "rupees",
                    "revenue",
                    "earning",
                    "earnings",
                ]
            ):
                metric = (
                    "export_revenue_lkr_bn"
                )

            if any(
                word in text
                for word in [
                    "usd",
                    "dollar",
                    "dollars",
                ]
            ):
                metric = (
                    "export_revenue_usd_mn"
                )

            if len(years) >= 2:
                return {
                    "operation": "compare_exports",
                    "parameters": {
                        "year_1": years[0],
                        "year_2": years[1],
                        "metric": metric,
                    },
                    "reason": (
                        "Two export years were found."
                    ),
                }

            if len(years) == 1:
                return {
                    "operation": "get_export",
                    "parameters": {
                        "year": years[0],
                        "metric": metric,
                    },
                    "reason": (
                        "One export year was found."
                    ),
                }

            if direction:
                return {
                    "operation": (
                        f"{direction}_export"
                    ),
                    "parameters": {
                        "metric": metric,
                    },
                    "reason": (
                        "An export extreme was requested."
                    ),
                }

            return {
                "operation": "export_trend",
                "parameters": {
                    "metric": metric,
                },
                "reason": (
                    "A general export trend was requested."
                ),
            }

        raise ValueError(
            "The selected route does not use "
            "the Data Agent."
        )

    def execute(
        self,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the selected pandas tool."""
        operation = plan["operation"]
        parameters = plan.get(
            "parameters",
            {},
        )

        if operation == "highest_annual":
            return self.repository.annual_extreme(
                "highest"
            )

        if operation == "lowest_annual":
            return self.repository.annual_extreme(
                "lowest"
            )

        if operation == "get_annual":
            return self.repository.annual_value(
                int(parameters["year"])
            )

        if operation == "compare_annual":
            return self.repository.compare_annual(
                int(parameters["year_1"]),
                int(parameters["year_2"]),
            )

        if operation == "annual_trend":
            return self.repository.annual_trend()

        if operation == "highest_month":
            return self.repository.monthly_extreme(
                "highest",
                self.optional_int(
                    parameters.get("year")
                ),
                parameters.get(
                    "metric",
                    "total_kg",
                ),
            )

        if operation == "lowest_month":
            return self.repository.monthly_extreme(
                "lowest",
                self.optional_int(
                    parameters.get("year")
                ),
                parameters.get(
                    "metric",
                    "total_kg",
                ),
            )

        if operation == "get_month":
            year = self.resolve_monthly_year(
                parameters.get("year")
            )

            return self.repository.monthly_value(
                year,
                str(parameters["month"]),
                parameters.get(
                    "metric",
                    "total_kg",
                ),
            )

        if operation == "compare_months":
            year = self.resolve_monthly_year(
                parameters.get("year")
            )

            return self.repository.compare_months(
                year,
                str(parameters["month_1"]),
                str(parameters["month_2"]),
                parameters.get(
                    "metric",
                    "total_kg",
                ),
            )

        if operation == "monthly_trend":
            return self.repository.monthly_trend(
                self.optional_int(
                    parameters.get("year")
                )
            )

        if operation == "highest_export":
            return self.repository.export_extreme(
                "highest",
                parameters.get(
                    "metric",
                    "export_volume_mn_kg",
                ),
            )

        if operation == "lowest_export":
            return self.repository.export_extreme(
                "lowest",
                parameters.get(
                    "metric",
                    "export_volume_mn_kg",
                ),
            )

        if operation == "get_export":
            return self.repository.export_value(
                int(parameters["year"]),
                parameters.get(
                    "metric",
                    "export_volume_mn_kg",
                ),
            )

        if operation == "compare_exports":
            return self.repository.compare_exports(
                int(parameters["year_1"]),
                int(parameters["year_2"]),
                parameters.get(
                    "metric",
                    "export_volume_mn_kg",
                ),
            )

        if operation == "export_trend":
            return self.repository.export_trend(
                parameters.get(
                    "metric",
                    "export_volume_mn_kg",
                )
            )

        raise ValueError(
            f"Unknown operation: {operation}"
        )

    def resolve_monthly_year(
        self,
        value: Any,
    ) -> int:
        """Use the only available year when the question omits it."""
        provided_year = self.optional_int(
            value
        )

        if provided_year is not None:
            return provided_year

        available_years = (
            self.repository
            .load_monthly()["year"]
            .unique()
            .tolist()
        )

        if len(available_years) != 1:
            raise DataValidationError(
                "Please include a year for the "
                "monthly question."
            )

        return int(available_years[0])

    @staticmethod
    def optional_int(
        value: Any,
    ) -> int | None:
        """Convert an optional value into an integer."""
        if value in (
            None,
            "",
            "null",
        ):
            return None

        return int(value)