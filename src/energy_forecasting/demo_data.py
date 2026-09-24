"""Prepare the two documented upload cases without a Streamlit dependency."""

from __future__ import annotations

import io
from datetime import date, datetime

from .aemo import NEM_TIME, profile_records, read_operational_demand_months
from .ausgrid import profile_daily_rows, read_customer_years
from .baseline import aemo_one_step_examples, ausgrid_next_slot_examples, partition_examples
from .sequences import aemo_history_windows, ausgrid_history_windows

EXPECTED_AEMO_NAMES = {
    f"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_{year}{month:02d}01.zip"
    for year, months in ((2025, range(8, 13)), (2026, range(1, 8)))
    for month in months
}


def prepare_uploaded_sources(aemo_files, ausgrid_file):
    """Validate these exact source cases and return causal chronological examples."""
    names = [item.name for item in aemo_files]
    if len(names) != 12 or len(set(names)) != 12 or set(names) != EXPECTED_AEMO_NAMES:
        missing = sorted(EXPECTED_AEMO_NAMES - set(names))
        extra = sorted(set(names) - EXPECTED_AEMO_NAMES)
        raise ValueError(f"Expected the documented 12 AEMO monthly ZIPs. Missing: {missing}; unexpected: {extra}")
    aemo_records = read_operational_demand_months(
        [io.BytesIO(item.getvalue()) for item in sorted(aemo_files, key=lambda file: file.name)], "NSW1"
    )
    aemo_profile = profile_records(aemo_records)
    if aemo_profile["non_30_minute_gaps"] or aemo_profile["missing_target_count"]:
        raise ValueError("AEMO source has gaps or missing targets; this fixed prototype does not impute them")
    if aemo_profile["count"] != 17520:
        raise ValueError("AEMO source coverage differs from the documented 12-month case")

    ausgrid_rows = read_customer_years(io.BytesIO(ausgrid_file.getvalue()), 1, "GG")
    ausgrid_profile = profile_daily_rows(ausgrid_rows)
    if (ausgrid_profile["day_count"] != 731 or ausgrid_profile["non_1_day_gaps"]
            or ausgrid_profile["blank_interval_count"] or ausgrid_profile["non_actual_row_count"]
            or ausgrid_profile["row_quality_not_provided_count"]):
        raise ValueError("Ausgrid customer 1 GG does not meet the documented consecutive, actual, no-blank rule")

    aemo_examples = aemo_history_windows(aemo_records, aemo_one_step_examples(aemo_records))
    ausgrid_examples = ausgrid_history_windows(ausgrid_rows, ausgrid_next_slot_examples(ausgrid_rows))
    return {
        "aemo": {
            "profile": aemo_profile,
            "parts": partition_examples(
                aemo_examples,
                datetime(2026, 4, 1, 4, 30, tzinfo=NEM_TIME),
                datetime(2026, 6, 1, 4, 30, tzinfo=NEM_TIME),
            ),
        },
        "ausgrid": {
            "profile": ausgrid_profile,
            "parts": partition_examples(
                ausgrid_examples, date(2013, 1, 1), date(2013, 4, 1), source_date_key=True
            ),
        },
    }
