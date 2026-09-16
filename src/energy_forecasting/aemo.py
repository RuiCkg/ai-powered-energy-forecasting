"""Read AEMO's nested monthly archive of daily operational-demand CSV files."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import io
from math import isfinite
from pathlib import Path
import zipfile


NEM_TIME = timezone(timedelta(hours=10))
EXPECTED_FIELDS = (
    "REGIONID",
    "INTERVAL_DATETIME",
    "OPERATIONAL_DEMAND",
    "OPERATIONAL_DEMAND_ADJUSTMENT",
    "WDR_ESTIMATE",
    "LASTCHANGED",
)


def read_operational_demand(archive_path: str | Path, region: str = "NSW1") -> list[dict]:
    """Return validated timestamp/target records from an AEMO ACTUAL_DAILY archive.

    The monthly archive contains daily ZIP files, which contain CSV files in
    AEMO's C/I/D interchange format. The target is measured operational demand
    in MW, not the forecast or an adjusted demand field.
    """

    records: list[dict] = []
    with zipfile.ZipFile(archive_path) as monthly:
        for daily_name in sorted(monthly.namelist()):
            if not daily_name.lower().endswith(".zip"):
                continue
            with zipfile.ZipFile(io.BytesIO(monthly.read(daily_name))) as daily:
                for csv_name in daily.namelist():
                    if not csv_name.lower().endswith(".csv"):
                        continue
                    text = daily.read(csv_name).decode("utf-8-sig", errors="replace")
                    header_seen = False
                    for fields in csv.reader(io.StringIO(text)):
                        if not fields:
                            continue
                        if fields[0] == "I" and fields[1:4] == ["OPERATIONAL_DEMAND", "ACTUAL", "3"]:
                            if tuple(fields[4:]) != EXPECTED_FIELDS:
                                raise ValueError(f"Unexpected AEMO columns in {csv_name}: {fields[4:]}")
                            header_seen = True
                        elif fields[0] == "D" and fields[1:4] == ["OPERATIONAL_DEMAND", "ACTUAL", "3"]:
                            if not header_seen or len(fields) != 10:
                                raise ValueError(f"Malformed AEMO demand row in {csv_name}")
                            if fields[4] != region:
                                continue
                            target_value = float(fields[6])
                            if not isfinite(target_value) or target_value < 0:
                                raise ValueError(f"Non-finite or negative AEMO demand in {csv_name}")
                            records.append(
                                {
                                    "timestamp": datetime.strptime(fields[5], "%Y/%m/%d %H:%M:%S").replace(tzinfo=NEM_TIME),
                                    "target": target_value,
                                    "region": fields[4],
                                    "unit": "MW",
                                }
                            )
                    if not header_seen:
                        raise ValueError(f"AEMO ACTUAL header was not found in {csv_name}")

    records.sort(key=lambda item: item["timestamp"])
    if not records:
        raise ValueError(f"No operational-demand records for region {region}")
    timestamps = [item["timestamp"] for item in records]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("Duplicate region/timestamp records in AEMO archive")
    return records


def read_operational_demand_months(archive_paths: list[str | Path], region: str = "NSW1") -> list[dict]:
    """Join monthly source archives, rejecting duplicate region timestamps."""

    if not archive_paths:
        raise ValueError("No monthly AEMO archives provided")
    records = [record for archive_path in archive_paths for record in read_operational_demand(archive_path, region)]
    records.sort(key=lambda item: item["timestamp"])
    timestamps = [record["timestamp"] for record in records]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("Duplicate region/timestamp records across AEMO monthly archives")
    return records


def profile_records(records: list[dict]) -> dict:
    """Summarise sampling and quality without imputing or changing source rows."""

    if not records:
        raise ValueError("No records to profile")
    timestamps = [record["timestamp"] for record in records]
    values = [record["target"] for record in records]
    expected_step = timedelta(minutes=30)
    minimum_index = min(range(len(records)), key=lambda index: values[index])
    largest_step_index = max(range(1, len(records)), key=lambda index: abs(values[index] - values[index - 1])) if len(records) > 1 else None
    return {
        "region": records[0]["region"],
        "unit": records[0]["unit"],
        "count": len(records),
        "first_timestamp": timestamps[0].isoformat(),
        "last_timestamp": timestamps[-1].isoformat(),
        "missing_target_count": sum(value is None for value in values),
        "non_30_minute_gaps": sum(
            right - left != expected_step for left, right in zip(timestamps, timestamps[1:])
        ),
        "minimum_MW": min(values),
        "minimum_MW_timestamp": timestamps[minimum_index].isoformat(),
        "maximum_MW": max(values),
        "mean_MW": sum(values) / len(values),
        "zero_MW_count": sum(value == 0 for value in values),
        "under_4000_MW_count_review_only": sum(value < 4000 for value in values),
        "largest_absolute_half_hour_step_MW": (
            abs(values[largest_step_index] - values[largest_step_index - 1]) if largest_step_index is not None else None
        ),
        "largest_step_target_timestamp": timestamps[largest_step_index].isoformat() if largest_step_index is not None else None,
    }
