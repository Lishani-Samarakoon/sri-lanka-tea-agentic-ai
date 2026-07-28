from pathlib import Path

import pandas as pd

from src.tools.data_tools import (
    DataRepository,
)


def create_repository(
    temporary_folder: Path,
) -> DataRepository:
    annual_path = (
        temporary_folder
        / "annual.csv"
    )

    monthly_path = (
        temporary_folder
        / "monthly.csv"
    )

    export_path = (
        temporary_folder
        / "exports.csv"
    )

    pd.DataFrame(
        [
            {
                "year": 2022,
                "production_mn_kg": 10.0,
                "source": "test",
            },
            {
                "year": 2023,
                "production_mn_kg": 12.0,
                "source": "test",
            },
        ]
    ).to_csv(
        annual_path,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "year": 2025,
                "month": "January",
                "high_kg": 2,
                "medium_kg": 3,
                "low_kg": 5,
                "total_kg": 10,
                "source": "test",
            },
            {
                "year": 2025,
                "month": "February",
                "high_kg": 3,
                "medium_kg": 4,
                "low_kg": 5,
                "total_kg": 12,
                "source": "test",
            },
        ]
    ).to_csv(
        monthly_path,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "year": 2022,
                "export_volume_mn_kg": 5.0,
                "export_revenue_lkr_bn": 8.0,
                "export_revenue_usd_mn": 20.0,
                "source": "test",
            },
            {
                "year": 2023,
                "export_volume_mn_kg": 6.0,
                "export_revenue_lkr_bn": 9.0,
                "export_revenue_usd_mn": 22.0,
                "source": "test",
            },
        ]
    ).to_csv(
        export_path,
        index=False,
    )

    return DataRepository(
        annual_path=annual_path,
        monthly_path=monthly_path,
        exports_path=export_path,
    )


def test_compare_annual(
    tmp_path: Path,
) -> None:
    repository = create_repository(
        tmp_path
    )

    result = repository.compare_annual(
        2022,
        2023,
    )

    assert (
        result["status"]
        == "success"
    )

    assert (
        "20.00%"
        in result["summary"]
    )


def test_highest_month(
    tmp_path: Path,
) -> None:
    repository = create_repository(
        tmp_path
    )

    result = repository.monthly_extreme(
        "highest",
        2025,
    )

    assert (
        result["records"][0]["month"]
        == "February"
    )


def test_compare_exports(
    tmp_path: Path,
) -> None:
    repository = create_repository(
        tmp_path
    )

    result = repository.compare_exports(
        2022,
        2023,
    )

    assert (
        result["status"]
        == "success"
    )

    assert (
        "20.00%"
        in result["summary"]
    )