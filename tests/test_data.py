"""Tests for common numerical data calculations."""

from src.tools.data_tools import (
    DataRepository,
)


def test_percentage_increase() -> None:
    """100 to 120 should be a 20% increase."""

    result = (
        DataRepository
        .percentage_change(
            100.0,
            120.0,
        )
    )

    assert result is not None
    assert round(result, 2) == 20.00


def test_percentage_decrease() -> None:
    """200 to 150 should be a 25% decrease."""

    result = (
        DataRepository
        .percentage_change(
            200.0,
            150.0,
        )
    )

    assert result is not None
    assert round(result, 2) == -25.00


def test_percentage_change_with_zero() -> None:
    """A percentage cannot be calculated from zero."""

    result = (
        DataRepository
        .percentage_change(
            0.0,
            100.0,
        )
    )

    assert result is None