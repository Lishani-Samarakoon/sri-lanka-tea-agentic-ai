import pytest

from src.tools.data_tools import (
    calculate_monthly_percentage_change,
    calculate_yearly_percentage_change,
    compare_two_months,
    compare_two_years,
    get_highest_production_month,
    get_highest_production_year,
    get_lowest_production_month,
    get_lowest_production_year,
    load_annual_data,
    load_monthly_data,
)


def test_load_annual_data():
    data = load_annual_data()

    assert len(data) == 7
    assert list(data.columns) == [
        "Year",
        "Production_mn_kg",
    ]


def test_load_monthly_data():
    data = load_monthly_data()

    assert len(data) == 12

    assert list(data.columns) == [
        "month",
        "high_kg",
        "medium_kg",
        "low_kg",
        "total_kg",
    ]


def test_highest_production_year():
    result = get_highest_production_year()

    assert result["year"] == 2010
    assert result["production_mn_kg"] == pytest.approx(
        331.4
    )


def test_lowest_production_year():
    result = get_lowest_production_year()

    assert result["year"] == 2009
    assert result["production_mn_kg"] == pytest.approx(
        291.0
    )


def test_compare_two_years():
    result = compare_two_years(2006, 2012)

    assert result["year_1"] == 2006
    assert result["year_2"] == 2012

    assert result["difference"] == pytest.approx(
        17.6
    )

    assert result["percentage_change"] == pytest.approx(
        5.66
    )


def test_yearly_percentage_change():
    result = calculate_yearly_percentage_change(
        2009,
        2010,
    )

    assert result["from_year"] == 2009
    assert result["to_year"] == 2010

    assert result["percentage_change"] == pytest.approx(
        13.88
    )


def test_highest_production_month():
    result = get_highest_production_month()

    assert result["month"] == "April"
    assert result["total_kg"] == 26466513


def test_lowest_production_month():
    result = get_lowest_production_month()

    assert result["month"] == "February"
    assert result["total_kg"] == 15757773


def test_compare_two_months():
    result = compare_two_months(
        "January",
        "February",
    )

    assert result["month_1"] == "January"
    assert result["month_2"] == "February"

    assert result["difference"] == -5868197

    assert result["percentage_change"] == pytest.approx(
        -27.13
    )


def test_monthly_percentage_change():
    result = calculate_monthly_percentage_change(
        "March",
        "April",
    )

    assert result["from_month"] == "March"
    assert result["to_month"] == "April"

    assert result["percentage_change"] == pytest.approx(
        6.82
    )


def test_invalid_year():
    with pytest.raises(ValueError):
        compare_two_years(
            2005,
            2012,
        )


def test_invalid_month():
    with pytest.raises(ValueError):
        compare_two_months(
            "WrongMonth",
            "January",
        )