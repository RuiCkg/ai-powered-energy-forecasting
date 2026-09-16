"""Read selected Ausgrid solar-home rows without inventing absolute timestamps."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime, timedelta
import io
from math import isfinite
from pathlib import Path
import zipfile


ANNUAL_FILES = (
    "Solar home 2010-2011.csv",
    "Solar home 2011-2012.csv",
    "Solar home 2012-2013.csv",
)
VALID_CATEGORIES = {"GG", "GC", "CL"}


def _parse_date(value: str):
    for date_format in ("%d-%b-%y", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            pass
    raise ValueError(f"Unsupported Ausgrid date: {value!r}")


def read_daily_rows(
    archive_path: str | Path, customer: int = 1, category: str = "GG", year_file: str = "Solar home 2012-2013.csv"
) -> list[dict]:
    """Read one customer/category from a wide annual CSV in the archive.

    The 48 readings stay in source order. Ausgrid's notes say the labels use
    local Eastern Standard or Daylight Savings Time. Absolute timestamps must
    not be assigned until a daylight-saving policy has been verified.
    """

    if year_file not in ANNUAL_FILES:
        raise ValueError(f"Unsupported annual file: {year_file}")
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Unsupported consumption category: {category}")
    result = []
    with zipfile.ZipFile(archive_path) as archive, archive.open(year_file) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline=""))
        next(reader)  # Source notice, not a table header.
        header = next(reader)
        if header[:5] != ["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date"]:
            raise ValueError(f"Unexpected Ausgrid header in {year_file}")
        if len(header[5:53]) != 48:
            raise ValueError("Expected 48 half-hour columns")
        for row in reader:
            if len(row) < 53 or not row[0].strip().isdigit():
                continue
            if int(row[0]) != customer or row[3].strip() != category:
                continue
            values = []
            for source_value in row[5:53]:
                if not source_value.strip():
                    values.append(None)
                    continue
                reading = float(source_value)
                if not isfinite(reading) or reading < 0:
                    raise ValueError(f"Non-finite or negative Ausgrid energy reading on {row[4]} for customer {customer}")
                values.append(reading)
            result.append(
                {
                    "customer": customer,
                    "category": category,
                    "date": _parse_date(row[4].strip()),
                    "clock_labels": tuple(header[5:53]),
                    "values_kWh": tuple(values),
                    "row_quality": row[53].strip() if len(row) > 53 else None,
                }
            )

    result.sort(key=lambda item: item["date"])
    if not result:
        raise ValueError(f"No Ausgrid rows for customer {customer}, category {category}, {year_file}")
    dates = [item["date"] for item in result]
    if len(dates) != len(set(dates)):
        raise ValueError("Duplicate Ausgrid customer/category/date rows")
    return result


def read_customer_years(
    archive_path: str | Path,
    customer: int = 1,
    category: str = "GG",
    year_files: tuple[str, ...] = ("Solar home 2011-2012.csv", "Solar home 2012-2013.csv"),
) -> list[dict]:
    """Join selected annual source rows, retaining date and 48 original slots."""

    if not year_files:
        raise ValueError("No Ausgrid annual files provided")
    rows = [row for year_file in year_files for row in read_daily_rows(archive_path, customer, category, year_file)]
    rows.sort(key=lambda item: item["date"])
    dates = [row["date"] for row in rows]
    if len(dates) != len(set(dates)):
        raise ValueError("Duplicate Ausgrid customer/category/date across annual files")
    return rows


def profile_daily_rows(rows: list[dict]) -> dict:
    """Describe a selected series without resolving DST or imputing values."""

    if not rows:
        raise ValueError("No rows to profile")
    readings = [value for row in rows for value in row["values_kWh"]]
    actual = [value for value in readings if value is not None]
    non_daily_gaps = sum((later["date"] - earlier["date"]).days != 1 for earlier, later in zip(rows, rows[1:]))
    return {
        "customer": rows[0]["customer"],
        "category": rows[0]["category"],
        "unit": "kWh per half-hour interval",
        "day_count": len(rows),
        "first_date": rows[0]["date"].isoformat(),
        "last_date": rows[-1]["date"].isoformat(),
        "non_1_day_gaps": non_daily_gaps,
        "interval_count": len(readings),
        "blank_interval_count": len(readings) - len(actual),
        "zero_interval_count": sum(value == 0 for value in actual),
        "non_actual_row_count": sum(row["row_quality"] == "NA" for row in rows),
        "row_quality_not_provided_count": sum(row["row_quality"] is None for row in rows),
        "minimum_kWh": min(actual) if actual else None,
        "maximum_kWh": max(actual) if actual else None,
        "absolute_timestamp_status": "not assigned: local daylight-saving policy pending",
    }


def profile_annual_archive(archive_path: str | Path, year_file: str = "Solar home 2012-2013.csv") -> dict:
    """Stream one annual CSV and summarize quality by customer and category.

    This is a source-format profile, not a normalized time series. Each source
    date has 48 slots, and DST is deliberately left unresolved.
    """

    if year_file not in ANNUAL_FILES:
        raise ValueError(f"Unsupported annual file: {year_file}")
    series = defaultdict(lambda: {"dates": set(), "rows": 0, "blank": 0, "non_actual": 0, "quality_unknown": 0})
    category_totals = defaultdict(lambda: {"rows": 0, "blank": 0, "non_actual": 0, "quality_unknown": 0})
    duplicate_rows = 0
    invalid_rows = 0
    with zipfile.ZipFile(archive_path) as archive, archive.open(year_file) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline=""))
        next(reader)
        header = next(reader)
        if header[:5] != ["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date"] or len(header[5:53]) != 48:
            raise ValueError(f"Unexpected Ausgrid header in {year_file}")
        has_quality = len(header) > 53 and header[53] == "Row Quality"
        for row in reader:
            if len(row) < 53 or not row[0].strip().isdigit() or row[3].strip() not in VALID_CATEGORIES:
                invalid_rows += 1
                continue
            try:
                source_date = _parse_date(row[4].strip())
                customer = int(row[0])
            except ValueError:
                invalid_rows += 1
                continue
            category = row[3].strip()
            item = series[(customer, category)]
            if source_date in item["dates"]:
                duplicate_rows += 1
            item["dates"].add(source_date)
            item["rows"] += 1
            blanks = sum(not value.strip() for value in row[5:53])
            quality = row[53].strip() if has_quality and len(row) > 53 else None
            non_actual = int(quality == "NA")
            unknown = int(quality is None)
            item["blank"] += blanks
            item["non_actual"] += non_actual
            item["quality_unknown"] += unknown
            totals = category_totals[category]
            totals["rows"] += 1
            totals["blank"] += blanks
            totals["non_actual"] += non_actual
            totals["quality_unknown"] += unknown

    gg = [(customer, item) for (customer, category), item in series.items() if category == "GG"]
    start_year = int(year_file.removesuffix(".csv").rsplit(" ", 1)[-1].split("-")[0])
    expected_dates = {date(start_year, 7, 1) + timedelta(days=offset) for offset in range((date(start_year + 1, 7, 1) - date(start_year, 7, 1)).days)}
    complete_gg = sorted(customer for customer, item in gg if item["rows"] == len(expected_dates) and item["dates"] == expected_dates and item["blank"] == 0 and item["non_actual"] == 0 and item["quality_unknown"] == 0)
    return {
        "year_file": year_file,
        "customer_count": len({customer for customer, _ in series}),
        "series_count": len(series),
        "category_totals": dict(sorted(category_totals.items())),
        "duplicate_customer_category_date_rows": duplicate_rows,
        "invalid_rows": invalid_rows,
        "gg_series_count": len(gg),
        "expected_gg_days_per_customer": len(expected_dates),
        "gg_missing_customer_days": sum(len(expected_dates - item["dates"]) for _, item in gg),
        "gg_full_year_no_blank_actual_customer_count": len(complete_gg),
        "gg_full_year_no_blank_actual_customer_examples": complete_gg[:10],
        "clock_policy": "48 source slots per day; daylight-saving timestamps not assigned",
    }
