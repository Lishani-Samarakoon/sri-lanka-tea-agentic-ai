from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

# data_tools.py is inside:
# project_folder/src/tools/data_tools.py
#
# parents[2] goes back to the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNUAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tea_annual_production.csv"
)

MONTHLY_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tea_monthly_production_2025.csv"
)

ANNUAL_SOURCE = "tea_annual_production.csv"
MONTHLY_SOURCE = "tea_monthly_production_2025.csv"

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


# ---------------------------------------------------------
# SMALL VALIDATION FUNCTION
# ---------------------------------------------------------

def validate_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """
    Check whether a dataset contains all required columns.
    """

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing these columns: "
            f"{sorted(missing_columns)}"
        )


# ---------------------------------------------------------
# LOAD ANNUAL DATA
# ---------------------------------------------------------

def load_annual_data() -> pd.DataFrame:
    """
    Read and validate the annual tea-production CSV file.
    """

    if not ANNUAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Annual dataset was not found here: "
            f"{ANNUAL_DATA_PATH}"
        )

    data = pd.read_csv(ANNUAL_DATA_PATH)

    required_columns = {
        "Year",
        "Production_mn_kg",
    }

    validate_columns(
        data,
        required_columns,
        "Annual dataset",
    )

    if data.empty:
        raise ValueError("Annual dataset is empty.")

    # Convert values into proper numerical types.
    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="raise",
    ).astype(int)

    data["Production_mn_kg"] = pd.to_numeric(
        data["Production_mn_kg"],
        errors="raise",
    )

    # Check for blank values.
    if data[list(required_columns)].isnull().any().any():
        raise ValueError(
            "Annual dataset contains blank values."
        )

    # A year should appear only once.
    if data["Year"].duplicated().any():
        raise ValueError(
            "Annual dataset contains duplicate years."
        )

    # Production should be greater than zero.
    if (data["Production_mn_kg"] <= 0).any():
        raise ValueError(
            "Annual production values must be greater than zero."
        )

    # Sort the table from oldest year to newest year.
    data = data.sort_values("Year").reset_index(drop=True)

    return data


# ---------------------------------------------------------
# LOAD MONTHLY DATA
# ---------------------------------------------------------

def load_monthly_data() -> pd.DataFrame:
    """
    Read and validate the monthly tea-production CSV file.
    """

    if not MONTHLY_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Monthly dataset was not found here: "
            f"{MONTHLY_DATA_PATH}"
        )

    data = pd.read_csv(MONTHLY_DATA_PATH)

    required_columns = {
        "month",
        "high_kg",
        "medium_kg",
        "low_kg",
        "total_kg",
    }

    validate_columns(
        data,
        required_columns,
        "Monthly dataset",
    )

    if data.empty:
        raise ValueError("Monthly dataset is empty.")

    # Remove accidental spaces and standardise month names.
    data["month"] = (
        data["month"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    numerical_columns = [
        "high_kg",
        "medium_kg",
        "low_kg",
        "total_kg",
    ]

    for column in numerical_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    # Check for blank cells.
    if data[list(required_columns)].isnull().any().any():
        raise ValueError(
            "Monthly dataset contains blank values."
        )

    # Check duplicate months.
    if data["month"].duplicated().any():
        raise ValueError(
            "Monthly dataset contains duplicate months."
        )

    months_in_file = set(data["month"])
    expected_months = set(MONTH_ORDER)

    missing_months = expected_months - months_in_file
    unexpected_months = months_in_file - expected_months

    if missing_months:
        raise ValueError(
            f"Monthly dataset is missing these months: "
            f"{sorted(missing_months)}"
        )

    if unexpected_months:
        raise ValueError(
            f"Monthly dataset contains invalid months: "
            f"{sorted(unexpected_months)}"
        )

    # All production values should be positive.
    if (data[numerical_columns] <= 0).any().any():
        raise ValueError(
            "Monthly production values must be greater than zero."
        )

    # Check:
    # high + medium + low should be close to the official total.
    #
    # The official PDF contains a one-kilogram difference
    # in June and October, so a difference of 1 kg is allowed.
    calculated_total = (
        data["high_kg"]
        + data["medium_kg"]
        + data["low_kg"]
    )

    total_difference = (
        data["total_kg"] - calculated_total
    ).abs()

    if (total_difference > 1).any():
        incorrect_rows = data.loc[
            total_difference > 1,
            ["month", "high_kg", "medium_kg", "low_kg", "total_kg"],
        ]

        raise ValueError(
            "Monthly totals contain differences greater than "
            f"1 kilogram:\n{incorrect_rows}"
        )

    # Sort January to December.
    month_positions = {
        month: position
        for position, month in enumerate(MONTH_ORDER)
    }

    data["_month_position"] = data["month"].map(
        month_positions
    )

    data = (
        data
        .sort_values("_month_position")
        .drop(columns="_month_position")
        .reset_index(drop=True)
    )

    return data


# ---------------------------------------------------------
# HELPER: FIND ONE YEAR
# ---------------------------------------------------------

def get_annual_row(year: int) -> pd.Series:
    """
    Find one year inside the annual dataset.
    """

    data = load_annual_data()

    try:
        selected_year = int(year)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Year must be a valid number."
        ) from error

    matching_rows = data[data["Year"] == selected_year]

    if matching_rows.empty:
        available_years = data["Year"].tolist()

        raise ValueError(
            f"Year {selected_year} is not available. "
            f"Available years: {available_years}"
        )

    return matching_rows.iloc[0]


# ---------------------------------------------------------
# HELPER: FIND ONE MONTH
# ---------------------------------------------------------

def get_monthly_row(month: str) -> pd.Series:
    """
    Find one month inside the monthly dataset.
    """

    if not isinstance(month, str) or not month.strip():
        raise ValueError(
            "Month must be entered as text."
        )

    data = load_monthly_data()

    selected_month = month.strip().title()

    matching_rows = data[
        data["month"] == selected_month
    ]

    if matching_rows.empty:
        raise ValueError(
            f"Month '{month}' is not valid. "
            "Enter a month from January to December."
        )

    return matching_rows.iloc[0]


# ---------------------------------------------------------
# HIGHEST ANNUAL PRODUCTION
# ---------------------------------------------------------

def get_highest_production_year() -> dict:
    """
    Return the year with the highest annual production.
    """

    data = load_annual_data()

    highest_index = data["Production_mn_kg"].idxmax()
    highest_row = data.loc[highest_index]

    return {
        "year": int(highest_row["Year"]),
        "production_mn_kg": float(
            highest_row["Production_mn_kg"]
        ),
        "unit": "million kilograms",
        "source": ANNUAL_SOURCE,
    }


# ---------------------------------------------------------
# LOWEST ANNUAL PRODUCTION
# ---------------------------------------------------------

def get_lowest_production_year() -> dict:
    """
    Return the year with the lowest annual production.
    """

    data = load_annual_data()

    lowest_index = data["Production_mn_kg"].idxmin()
    lowest_row = data.loc[lowest_index]

    return {
        "year": int(lowest_row["Year"]),
        "production_mn_kg": float(
            lowest_row["Production_mn_kg"]
        ),
        "unit": "million kilograms",
        "source": ANNUAL_SOURCE,
    }


# ---------------------------------------------------------
# COMPARE TWO YEARS
# ---------------------------------------------------------

def compare_two_years(
    year1: int,
    year2: int,
) -> dict:
    """
    Compare tea production between two years.
    """

    first_row = get_annual_row(year1)
    second_row = get_annual_row(year2)

    value1 = float(first_row["Production_mn_kg"])
    value2 = float(second_row["Production_mn_kg"])

    difference = value2 - value1

    if value1 == 0:
        percentage_change = None
    else:
        percentage_change = (
            difference / value1
        ) * 100

    return {
        "year_1": int(first_row["Year"]),
        "value_1": value1,
        "year_2": int(second_row["Year"]),
        "value_2": value2,
        "difference": round(difference, 2),
        "percentage_change": (
            round(percentage_change, 2)
            if percentage_change is not None
            else None
        ),
        "unit": "million kilograms",
        "source": ANNUAL_SOURCE,
    }


# ---------------------------------------------------------
# YEARLY PERCENTAGE CHANGE
# ---------------------------------------------------------

def calculate_yearly_percentage_change(
    year1: int,
    year2: int,
) -> dict:
    """
    Calculate the percentage change from year1 to year2.
    """

    comparison = compare_two_years(year1, year2)

    return {
        "from_year": comparison["year_1"],
        "from_value": comparison["value_1"],
        "to_year": comparison["year_2"],
        "to_value": comparison["value_2"],
        "percentage_change": comparison[
            "percentage_change"
        ],
        "unit": comparison["unit"],
        "source": comparison["source"],
    }


# ---------------------------------------------------------
# HIGHEST MONTHLY PRODUCTION
# ---------------------------------------------------------

def get_highest_production_month() -> dict:
    """
    Return the month with the highest total production.
    """

    data = load_monthly_data()

    highest_index = data["total_kg"].idxmax()
    highest_row = data.loc[highest_index]

    return {
        "month": str(highest_row["month"]),
        "total_kg": int(highest_row["total_kg"]),
        "high_kg": int(highest_row["high_kg"]),
        "medium_kg": int(highest_row["medium_kg"]),
        "low_kg": int(highest_row["low_kg"]),
        "unit": "kilograms",
        "source": MONTHLY_SOURCE,
    }


# ---------------------------------------------------------
# LOWEST MONTHLY PRODUCTION
# ---------------------------------------------------------

def get_lowest_production_month() -> dict:
    """
    Return the month with the lowest total production.
    """

    data = load_monthly_data()

    lowest_index = data["total_kg"].idxmin()
    lowest_row = data.loc[lowest_index]

    return {
        "month": str(lowest_row["month"]),
        "total_kg": int(lowest_row["total_kg"]),
        "high_kg": int(lowest_row["high_kg"]),
        "medium_kg": int(lowest_row["medium_kg"]),
        "low_kg": int(lowest_row["low_kg"]),
        "unit": "kilograms",
        "source": MONTHLY_SOURCE,
    }


# ---------------------------------------------------------
# COMPARE TWO MONTHS
# ---------------------------------------------------------

def compare_two_months(
    month1: str,
    month2: str,
) -> dict:
    """
    Compare total tea production between two months.
    """

    first_row = get_monthly_row(month1)
    second_row = get_monthly_row(month2)

    value1 = int(first_row["total_kg"])
    value2 = int(second_row["total_kg"])

    difference = value2 - value1

    if value1 == 0:
        percentage_change = None
    else:
        percentage_change = (
            difference / value1
        ) * 100

    return {
        "month_1": str(first_row["month"]),
        "value_1": value1,
        "month_2": str(second_row["month"]),
        "value_2": value2,
        "difference": difference,
        "percentage_change": (
            round(percentage_change, 2)
            if percentage_change is not None
            else None
        ),
        "unit": "kilograms",
        "source": MONTHLY_SOURCE,
    }


# ---------------------------------------------------------
# MONTHLY PERCENTAGE CHANGE
# ---------------------------------------------------------

def calculate_monthly_percentage_change(
    month1: str,
    month2: str,
) -> dict:
    """
    Calculate percentage change from month1 to month2.
    """

    comparison = compare_two_months(
        month1,
        month2,
    )

    return {
        "from_month": comparison["month_1"],
        "from_value": comparison["value_1"],
        "to_month": comparison["month_2"],
        "to_value": comparison["value_2"],
        "percentage_change": comparison[
            "percentage_change"
        ],
        "unit": comparison["unit"],
        "source": comparison["source"],
    }


# ---------------------------------------------------------
# SIMPLE MANUAL TEST
# ---------------------------------------------------------

if __name__ == "__main__":
    print("\nANNUAL DATA")
    print(load_annual_data())

    print("\nMONTHLY DATA")
    print(load_monthly_data())

    print("\nHIGHEST YEAR")
    print(get_highest_production_year())

    print("\nLOWEST YEAR")
    print(get_lowest_production_year())

    print("\nCOMPARE 2006 AND 2012")
    print(compare_two_years(2006, 2012))

    print("\nYEARLY PERCENTAGE CHANGE: 2009 TO 2010")
    print(
        calculate_yearly_percentage_change(
            2009,
            2010,
        )
    )

    print("\nHIGHEST MONTH")
    print(get_highest_production_month())

    print("\nLOWEST MONTH")
    print(get_lowest_production_month())

    print("\nCOMPARE JANUARY AND FEBRUARY")
    print(
        compare_two_months(
            "January",
            "February",
        )
    )

    print("\nMONTHLY PERCENTAGE CHANGE: MARCH TO APRIL")
    print(
        calculate_monthly_percentage_change(
            "March",
            "April",
        )
    )