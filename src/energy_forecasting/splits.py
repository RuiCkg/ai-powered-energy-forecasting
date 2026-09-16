"""Chronological source-record splits; no look-ahead features or windows yet."""

from __future__ import annotations

from datetime import date, datetime, timedelta


def _split_sorted(records: list[dict], key: str, validation_start, test_start) -> dict[str, list[dict]]:
    if not records:
        raise ValueError("No source records to split")
    values = [record[key] for record in records]
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError("Source records must have unique, chronological keys")
    if not values[0] < validation_start < test_start <= values[-1]:
        raise ValueError("Split boundaries must fall inside source coverage in order")
    parts = {
        "train": [record for record in records if record[key] < validation_start],
        "validation": [record for record in records if validation_start <= record[key] < test_start],
        "test": [record for record in records if record[key] >= test_start],
    }
    if any(not part for part in parts.values()):
        raise ValueError("Each chronological split must have source records")
    return parts


def split_aemo_records(
    records: list[dict], validation_start: datetime, test_start: datetime
) -> dict[str, list[dict]]:
    """Split fixed-UTC+10 AEMO intervals by exact timestamp."""

    if validation_start.tzinfo is None or test_start.tzinfo is None:
        raise ValueError("AEMO boundaries must carry the fixed NEM timezone")
    if validation_start.utcoffset() != timedelta(hours=10) or test_start.utcoffset() != timedelta(hours=10):
        raise ValueError("AEMO boundaries must use the fixed NEM UTC+10 offset")
    return _split_sorted(records, "timestamp", validation_start, test_start)


def split_ausgrid_days(
    rows: list[dict], validation_start: date, test_start: date
) -> dict[str, list[dict]]:
    """Split whole Ausgrid source days before resolving DST clock labels."""

    if isinstance(validation_start, datetime) or isinstance(test_start, datetime):
        raise ValueError("Ausgrid boundaries must be source dates, not timestamps")
    return _split_sorted(rows, "date", validation_start, test_start)
