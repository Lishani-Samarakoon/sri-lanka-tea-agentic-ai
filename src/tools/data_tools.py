"""Pandas tools for official tea production and export datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    ANNUAL_EXPORTS_CSV,
    ANNUAL_PRODUCTION_CSV,
    MONTHLY_PRODUCTION_CSV,
)
from src.schemas import json_safe


MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


class DataValidationError(ValueError):
    """Raised when an official dataset cannot be used safely."""


@dataclass
class DataRepository:
    """Loads datasets and performs deterministic calculations."""

    annual_path: Path = ANNUAL_PRODUCTION_CSV
    monthly_path: Path = MONTHLY_PRODUCTION_CSV
    exports_path: Path = ANNUAL_EXPORTS_CSV

    @staticmethod
    def percentage_change(
        old_value: float,
        new_value: float,
    ) -> float | None:
        """Calculate percentage change."""
        if old_value == 0:
            return None

        return (
            (new_value - old_value)
            / old_value
        ) * 100

    @staticmethod
    def read_csv(
        path: Path,
        required_columns: set[str],
    ) -> pd.DataFrame:
        """Read and validate a CSV file."""
        if not path.exists():
            raise DataValidationError(
                f"Dataset not found: {path}"
            )

        try:
            dataframe = pd.read_csv(path)

        except pd.errors.EmptyDataError as error:
            raise DataValidationError(
                f"Dataset is empty: {path}"
            ) from error

        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            missing_text = ", ".join(
                sorted(missing_columns)
            )

            raise DataValidationError(
                f"{path.name} is missing columns: "
                f"{missing_text}"
            )

        if dataframe.empty:
            raise DataValidationError(
                f"{path.name} contains only headings. "
                "Add verified official data rows."
            )

        return dataframe

    def load_annual(self) -> pd.DataFrame:
        """Load annual production data."""
        dataframe = self.read_csv(
            self.annual_path,
            {
                "year",
                "production_mn_kg",
                "source",
            },
        ).copy()

        dataframe["year"] = pd.to_numeric(
            dataframe["year"],
            errors="raise",
        ).astype(int)

        dataframe["production_mn_kg"] = pd.to_numeric(
            dataframe["production_mn_kg"],
            errors="raise",
        )

        return dataframe.sort_values(
            "year"
        ).reset_index(drop=True)

    def load_monthly(self) -> pd.DataFrame:
        """Load monthly tea-production data."""
        dataframe = self.read_csv(
            self.monthly_path,
            {
                "month",
                "high_kg",
                "medium_kg",
                "low_kg",
                "total_kg",
                "source",
            },
        ).copy()

        # Your 2025 file may not have a year column.
        if "year" not in dataframe.columns:
            year_match = re.search(
                r"(?:19|20)\d{2}",
                self.monthly_path.name,
            )

            inferred_year = (
                int(year_match.group())
                if year_match
                else 2025
            )

            dataframe["year"] = inferred_year

        dataframe["year"] = pd.to_numeric(
            dataframe["year"],
            errors="raise",
        ).astype(int)

        number_columns = [
            "high_kg",
            "medium_kg",
            "low_kg",
            "total_kg",
        ]

        for column in number_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="raise",
            )

        dataframe["month"] = (
            dataframe["month"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        invalid_months = sorted(
            set(dataframe["month"])
            - set(MONTH_ORDER)
        )

        if invalid_months:
            raise DataValidationError(
                "Invalid month names: "
                + ", ".join(invalid_months)
            )

        calculated_total = (
            dataframe["high_kg"]
            + dataframe["medium_kg"]
            + dataframe["low_kg"]
        )

        incorrect_total = (
            calculated_total
            - dataframe["total_kg"]
        ).abs() > 1

        if incorrect_total.any():
            csv_rows = (
                dataframe.index[incorrect_total] + 2
            ).tolist()

            raise DataValidationError(
                "total_kg must equal high_kg + "
                "medium_kg + low_kg. Check CSV rows: "
                f"{csv_rows}"
            )

        month_numbers = {
            month: position + 1
            for position, month
            in enumerate(MONTH_ORDER)
        }

        dataframe["month_number"] = (
            dataframe["month"]
            .map(month_numbers)
        )

        return dataframe.sort_values(
            ["year", "month_number"]
        ).reset_index(drop=True)

    def load_exports(self) -> pd.DataFrame:
        """Load annual export data."""
        dataframe = self.read_csv(
            self.exports_path,
            {
                "year",
                "export_volume_mn_kg",
                "export_revenue_lkr_bn",
                "source",
            },
        ).copy()

        if (
            "export_revenue_usd_mn"
            not in dataframe.columns
        ):
            if (
                "export_revenue_usd_bn"
                in dataframe.columns
            ):
                dataframe[
                    "export_revenue_usd_mn"
                ] = (
                    pd.to_numeric(
                        dataframe[
                            "export_revenue_usd_bn"
                        ],
                        errors="raise",
                    )
                    * 1000
                )

            else:
                raise DataValidationError(
                    f"{self.exports_path.name} must "
                    "contain export_revenue_usd_mn "
                    "or export_revenue_usd_bn."
                )

        dataframe["year"] = pd.to_numeric(
            dataframe["year"],
            errors="raise",
        ).astype(int)

        number_columns = [
            "export_volume_mn_kg",
            "export_revenue_lkr_bn",
            "export_revenue_usd_mn",
        ]

        for column in number_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="raise",
            )

        return dataframe.sort_values(
            "year"
        ).reset_index(drop=True)

    @staticmethod
    def create_result(
        *,
        operation: str,
        summary: str,
        records: list[dict[str, Any]],
        sources: list[str],
        chart: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a standard result returned by every tool."""
        return json_safe(
            {
                "status": "success",
                "operation": operation,
                "summary": summary,
                "records": records,
                "sources": sorted(
                    set(sources)
                ),
                "chart": chart,
            }
        )

    def annual_extreme(
        self,
        direction: str,
    ) -> dict[str, Any]:
        """Find the highest or lowest annual production."""
        dataframe = self.load_annual()

        if direction == "highest":
            row_index = dataframe[
                "production_mn_kg"
            ].idxmax()

        elif direction == "lowest":
            row_index = dataframe[
                "production_mn_kg"
            ].idxmin()

        else:
            raise DataValidationError(
                "Direction must be highest or lowest."
            )

        row = dataframe.loc[row_index]

        summary = (
            f"The {direction} annual tea production "
            f"in the dataset was "
            f"{row['production_mn_kg']:,.2f} million kg "
            f"in {int(row['year'])}."
        )

        record = row[
            [
                "year",
                "production_mn_kg",
                "source",
            ]
        ].to_dict()

        return self.create_result(
            operation=(
                f"{direction}_annual_production"
            ),
            summary=summary,
            records=[record],
            sources=[str(row["source"])],
        )

    def annual_value(
        self,
        year: int,
    ) -> dict[str, Any]:
        """Get annual production for one year."""
        dataframe = self.load_annual()

        result = dataframe[
            dataframe["year"] == year
        ]

        if result.empty:
            raise DataValidationError(
                "Annual production data is not "
                f"available for {year}."
            )

        row = result.iloc[0]

        summary = (
            f"Tea production in {year} was "
            f"{row['production_mn_kg']:,.2f} "
            "million kg."
        )

        record = row[
            [
                "year",
                "production_mn_kg",
                "source",
            ]
        ].to_dict()

        return self.create_result(
            operation="get_annual_production",
            summary=summary,
            records=[record],
            sources=[str(row["source"])],
        )

    def compare_annual(
        self,
        year_1: int,
        year_2: int,
    ) -> dict[str, Any]:
        """Compare annual tea production between two years."""
        dataframe = self.load_annual()

        result = dataframe[
            dataframe["year"].isin(
                [year_1, year_2]
            )
        ]

        found_years = set(
            result["year"].astype(int)
        )

        missing_years = [
            year
            for year in [year_1, year_2]
            if year not in found_years
        ]

        if missing_years:
            raise DataValidationError(
                "Annual production data is not "
                "available for: "
                + ", ".join(
                    map(str, missing_years)
                )
            )

        first_row = result[
            result["year"] == year_1
        ].iloc[0]

        second_row = result[
            result["year"] == year_2
        ].iloc[0]

        old_value = float(
            first_row["production_mn_kg"]
        )

        new_value = float(
            second_row["production_mn_kg"]
        )

        difference = new_value - old_value

        percentage = self.percentage_change(
            old_value,
            new_value,
        )

        if difference > 0:
            direction = "increased"

        elif difference < 0:
            direction = "decreased"

        else:
            direction = "did not change"

        percentage_text = (
            "undefined"
            if percentage is None
            else f"{abs(percentage):.2f}%"
        )

        summary = (
            f"Tea production {direction} from "
            f"{old_value:,.2f} million kg in "
            f"{year_1} to {new_value:,.2f} "
            f"million kg in {year_2}. "
            f"The absolute change was "
            f"{difference:,.2f} million kg. "
            f"The percentage change was "
            f"{percentage_text}."
        )

        records = result[
            [
                "year",
                "production_mn_kg",
                "source",
            ]
        ].sort_values("year").to_dict(
            "records"
        )

        return self.create_result(
            operation="compare_annual_production",
            summary=summary,
            records=records,
            sources=result[
                "source"
            ].astype(str).tolist(),
            chart={
                "type": "bar",
                "x": "year",
                "y": "production_mn_kg",
                "data": records,
                "y_label": (
                    "Production (million kg)"
                ),
            },
        )

    def annual_trend(self) -> dict[str, Any]:
        """Return all annual production values."""
        dataframe = self.load_annual()

        records = dataframe[
            [
                "year",
                "production_mn_kg",
                "source",
            ]
        ].to_dict("records")

        summary = (
            f"The annual dataset contains "
            f"{len(dataframe)} records from "
            f"{int(dataframe['year'].min())} to "
            f"{int(dataframe['year'].max())}."
        )

        return self.create_result(
            operation="annual_production_trend",
            summary=summary,
            records=records,
            sources=dataframe[
                "source"
            ].astype(str).tolist(),
            chart={
                "type": "line",
                "x": "year",
                "y": "production_mn_kg",
                "data": records,
                "y_label": (
                    "Production (million kg)"
                ),
            },
        )

    def monthly_extreme(
        self,
        direction: str,
        year: int | None = None,
        metric: str = "total_kg",
    ) -> dict[str, Any]:
        """Find the highest or lowest month."""
        allowed_metrics = {
            "high_kg",
            "medium_kg",
            "low_kg",
            "total_kg",
        }

        if metric not in allowed_metrics:
            raise DataValidationError(
                f"Unsupported monthly metric: {metric}"
            )

        dataframe = self.load_monthly()

        if year is not None:
            dataframe = dataframe[
                dataframe["year"] == year
            ]

            if dataframe.empty:
                raise DataValidationError(
                    "Monthly production data is "
                    f"not available for {year}."
                )

        if direction == "highest":
            row_index = dataframe[
                metric
            ].idxmax()

        elif direction == "lowest":
            row_index = dataframe[
                metric
            ].idxmin()

        else:
            raise DataValidationError(
                "Direction must be highest or lowest."
            )

        row = dataframe.loc[row_index]

        summary = (
            f"The {direction} "
            f"{metric.replace('_', ' ')} was "
            f"{row[metric]:,.0f} kg in "
            f"{row['month']} {int(row['year'])}."
        )

        columns = [
            "year",
            "month",
            "high_kg",
            "medium_kg",
            "low_kg",
            "total_kg",
            "source",
        ]

        return self.create_result(
            operation=(
                f"{direction}_monthly_production"
            ),
            summary=summary,
            records=[
                row[columns].to_dict()
            ],
            sources=[str(row["source"])],
        )

    def monthly_value(
        self,
        year: int,
        month: str,
        metric: str = "total_kg",
    ) -> dict[str, Any]:
        """Return production for one month."""
        allowed_metrics = {
            "high_kg",
            "medium_kg",
            "low_kg",
            "total_kg",
        }

        if metric not in allowed_metrics:
            raise DataValidationError(
                f"Unsupported monthly metric: {metric}"
            )

        month = month.strip().title()

        dataframe = self.load_monthly()

        result = dataframe[
            (dataframe["year"] == year)
            & (dataframe["month"] == month)
        ]

        if result.empty:
            raise DataValidationError(
                "Monthly production data is not "
                f"available for {month} {year}."
            )

        row = result.iloc[0]

        summary = (
            f"{metric.replace('_', ' ').title()} "
            f"in {month} {year} was "
            f"{row[metric]:,.0f} kg."
        )

        columns = [
            "year",
            "month",
            "high_kg",
            "medium_kg",
            "low_kg",
            "total_kg",
            "source",
        ]

        return self.create_result(
            operation="get_monthly_production",
            summary=summary,
            records=[
                row[columns].to_dict()
            ],
            sources=[str(row["source"])],
        )

    def compare_months(
        self,
        year: int,
        month_1: str,
        month_2: str,
        metric: str = "total_kg",
    ) -> dict[str, Any]:
        """Compare production between two months."""
        allowed_metrics = {
            "high_kg",
            "medium_kg",
            "low_kg",
            "total_kg",
        }

        if metric not in allowed_metrics:
            raise DataValidationError(
                f"Unsupported monthly metric: {metric}"
            )

        month_1 = month_1.strip().title()
        month_2 = month_2.strip().title()

        dataframe = self.load_monthly()

        result = dataframe[
            (dataframe["year"] == year)
            & dataframe["month"].isin(
                [month_1, month_2]
            )
        ]

        found_months = set(
            result["month"]
        )

        missing_months = [
            month
            for month in [month_1, month_2]
            if month not in found_months
        ]

        if missing_months:
            raise DataValidationError(
                f"Monthly data for {year} is not "
                "available for: "
                + ", ".join(missing_months)
            )

        first_row = result[
            result["month"] == month_1
        ].iloc[0]

        second_row = result[
            result["month"] == month_2
        ].iloc[0]

        old_value = float(first_row[metric])
        new_value = float(second_row[metric])

        difference = new_value - old_value

        percentage = self.percentage_change(
            old_value,
            new_value,
        )

        if difference > 0:
            direction = "increased"

        elif difference < 0:
            direction = "decreased"

        else:
            direction = "did not change"

        percentage_text = (
            "undefined"
            if percentage is None
            else f"{abs(percentage):.2f}%"
        )

        summary = (
            f"{metric.replace('_', ' ').title()} "
            f"{direction} from {old_value:,.0f} kg "
            f"in {month_1} {year} to "
            f"{new_value:,.0f} kg in "
            f"{month_2} {year}. "
            f"The change was {difference:,.0f} kg "
            f"({percentage_text})."
        )

        columns = [
            "year",
            "month",
            "high_kg",
            "medium_kg",
            "low_kg",
            "total_kg",
            "source",
        ]

        records = result.sort_values(
            "month_number"
        )[columns].to_dict("records")

        return self.create_result(
            operation="compare_monthly_production",
            summary=summary,
            records=records,
            sources=result[
                "source"
            ].astype(str).tolist(),
            chart={
                "type": "bar",
                "x": "month",
                "y": metric,
                "data": records,
                "y_label": (
                    metric.replace("_", " ").title()
                ),
            },
        )

    def monthly_trend(
        self,
        year: int | None = None,
    ) -> dict[str, Any]:
        """Return monthly production records."""
        dataframe = self.load_monthly()

        if year is not None:
            dataframe = dataframe[
                dataframe["year"] == year
            ]

            if dataframe.empty:
                raise DataValidationError(
                    "Monthly data is not available "
                    f"for {year}."
                )

        columns = [
            "year",
            "month",
            "high_kg",
            "medium_kg",
            "low_kg",
            "total_kg",
            "source",
        ]

        records = dataframe[
            columns
        ].to_dict("records")

        years = sorted(
            dataframe["year"]
            .unique()
            .tolist()
        )

        summary = (
            f"The dataset contains {len(dataframe)} "
            f"monthly records for year(s): {years}."
        )

        return self.create_result(
            operation="monthly_production_trend",
            summary=summary,
            records=records,
            sources=dataframe[
                "source"
            ].astype(str).tolist(),
            chart={
                "type": "line",
                "x": "month",
                "y": "total_kg",
                "data": records,
                "y_label": (
                    "Total production (kg)"
                ),
            },
        )

    @staticmethod
    def export_unit(metric: str) -> str:
        """Return the correct export unit."""
        units = {
            "export_volume_mn_kg": "million kg",
            "export_revenue_lkr_bn": "LKR billion",
            "export_revenue_usd_mn": "USD million",
        }

        if metric not in units:
            raise DataValidationError(
                f"Unsupported export metric: {metric}"
            )

        return units[metric]

    def export_extreme(
        self,
        direction: str,
        metric: str = "export_volume_mn_kg",
    ) -> dict[str, Any]:
        """Find the highest or lowest export value."""
        unit = self.export_unit(metric)
        dataframe = self.load_exports()

        if direction == "highest":
            row_index = dataframe[
                metric
            ].idxmax()

        elif direction == "lowest":
            row_index = dataframe[
                metric
            ].idxmin()

        else:
            raise DataValidationError(
                "Direction must be highest or lowest."
            )

        row = dataframe.loc[row_index]

        summary = (
            f"The {direction} "
            f"{metric.replace('_', ' ')} was "
            f"{row[metric]:,.2f} {unit} in "
            f"{int(row['year'])}."
        )

        columns = [
            "year",
            "export_volume_mn_kg",
            "export_revenue_lkr_bn",
            "export_revenue_usd_mn",
            "source",
        ]

        return self.create_result(
            operation=f"{direction}_export_metric",
            summary=summary,
            records=[
                row[columns].to_dict()
            ],
            sources=[str(row["source"])],
        )

    def export_value(
        self,
        year: int,
        metric: str = "export_volume_mn_kg",
    ) -> dict[str, Any]:
        """Get an export value for one year."""
        unit = self.export_unit(metric)
        dataframe = self.load_exports()

        result = dataframe[
            dataframe["year"] == year
        ]

        if result.empty:
            raise DataValidationError(
                f"Export data is not available for {year}."
            )

        row = result.iloc[0]

        summary = (
            f"{metric.replace('_', ' ').title()} "
            f"in {year} was "
            f"{row[metric]:,.2f} {unit}."
        )

        columns = [
            "year",
            "export_volume_mn_kg",
            "export_revenue_lkr_bn",
            "export_revenue_usd_mn",
            "source",
        ]

        return self.create_result(
            operation="get_export_metric",
            summary=summary,
            records=[
                row[columns].to_dict()
            ],
            sources=[str(row["source"])],
        )

    def compare_exports(
        self,
        year_1: int,
        year_2: int,
        metric: str = "export_volume_mn_kg",
    ) -> dict[str, Any]:
        """Compare an export metric between two years."""
        unit = self.export_unit(metric)
        dataframe = self.load_exports()

        result = dataframe[
            dataframe["year"].isin(
                [year_1, year_2]
            )
        ]

        found_years = set(
            result["year"].astype(int)
        )

        missing_years = [
            year
            for year in [year_1, year_2]
            if year not in found_years
        ]

        if missing_years:
            raise DataValidationError(
                "Export data is not available for: "
                + ", ".join(
                    map(str, missing_years)
                )
            )

        first_row = result[
            result["year"] == year_1
        ].iloc[0]

        second_row = result[
            result["year"] == year_2
        ].iloc[0]

        old_value = float(first_row[metric])
        new_value = float(second_row[metric])

        difference = new_value - old_value

        percentage = self.percentage_change(
            old_value,
            new_value,
        )

        if difference > 0:
            direction = "increased"

        elif difference < 0:
            direction = "decreased"

        else:
            direction = "did not change"

        percentage_text = (
            "undefined"
            if percentage is None
            else f"{abs(percentage):.2f}%"
        )

        summary = (
            f"{metric.replace('_', ' ').title()} "
            f"{direction} from "
            f"{old_value:,.2f} {unit} in {year_1} "
            f"to {new_value:,.2f} {unit} in "
            f"{year_2}. The change was "
            f"{difference:,.2f} {unit} "
            f"({percentage_text})."
        )

        columns = [
            "year",
            "export_volume_mn_kg",
            "export_revenue_lkr_bn",
            "export_revenue_usd_mn",
            "source",
        ]

        records = result[
            columns
        ].sort_values("year").to_dict(
            "records"
        )

        return self.create_result(
            operation="compare_export_metric",
            summary=summary,
            records=records,
            sources=result[
                "source"
            ].astype(str).tolist(),
            chart={
                "type": "bar",
                "x": "year",
                "y": metric,
                "data": records,
                "y_label": (
                    f"{metric.replace('_', ' ').title()} "
                    f"({unit})"
                ),
            },
        )

    def export_trend(
        self,
        metric: str = "export_volume_mn_kg",
    ) -> dict[str, Any]:
        """Return all annual export data."""
        self.export_unit(metric)
        dataframe = self.load_exports()

        columns = [
            "year",
            "export_volume_mn_kg",
            "export_revenue_lkr_bn",
            "export_revenue_usd_mn",
            "source",
        ]

        records = dataframe[
            columns
        ].to_dict("records")

        summary = (
            f"The export dataset contains "
            f"{len(dataframe)} records from "
            f"{int(dataframe['year'].min())} to "
            f"{int(dataframe['year'].max())}."
        )

        return self.create_result(
            operation="export_trend",
            summary=summary,
            records=records,
            sources=dataframe[
                "source"
            ].astype(str).tolist(),
            chart={
                "type": "line",
                "x": "year",
                "y": metric,
                "data": records,
                "y_label": (
                    metric.replace("_", " ").title()
                ),
            },
        )