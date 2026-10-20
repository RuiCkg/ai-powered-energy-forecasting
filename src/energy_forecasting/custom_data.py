"""Validated adapter for an approved-schema custom energy CSV."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import io
import math
from pathlib import Path

import pandas as pd


MAX_CUSTOM_CSV_BYTES = 20 * 1024 * 1024
MIN_PART_EXAMPLES = 10


def build_synthetic_demo_csv(row_count: int = 672) -> bytes:
    """Return a valid, deterministic two-week half-hourly demonstration CSV."""

    minimum_rows = 48 + 3 * MIN_PART_EXAMPLES
    if row_count < minimum_rows:
        raise ValueError(f"The demonstration CSV needs at least {minimum_rows} rows")
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00+10:00")
    for index in range(row_count):
        timestamp = start + timedelta(minutes=30 * index)
        slot = index % 48
        day = index // 48
        daylight = max(0.0, math.sin(math.pi * (slot - 12) / 24))
        morning_peak = math.exp(-((slot - 15) / 5) ** 2)
        evening_peak = math.exp(-((slot - 37) / 6) ** 2)
        weekend_adjustment = -0.12 if day % 7 in {5, 6} else 0.08
        temperature = 19 + 7 * math.sin(2 * math.pi * (slot - 14) / 48)
        energy = max(
            0.05,
            0.55
            + 0.52 * morning_peak
            + 0.82 * evening_peak
            + 0.18 * daylight
            + weekend_adjustment,
        )
        rows.append({
            "timestamp": timestamp.isoformat(),
            "energy_kwh": round(energy, 4),
            "temperature_c": round(temperature, 2),
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


@dataclass(frozen=True)
class CustomCSVInspection:
    """Safe metadata needed to configure a custom CSV import."""

    columns: tuple[str, ...]
    row_count: int
    byte_size: int
    sha256: str


def _validated_bytes(name: str, data: bytes) -> bytes:
    safe_name = Path(name).name
    if not safe_name or safe_name != name or not safe_name.lower().endswith(".csv"):
        raise ValueError("Upload one file whose name ends in .csv")
    if not data:
        raise ValueError("The CSV file is empty")
    if len(data) > MAX_CUSTOM_CSV_BYTES:
        raise ValueError("The CSV file is larger than the 20 MB prototype limit")
    if b"\x00" in data:
        raise ValueError("The CSV contains null bytes and cannot be treated as text")
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("The CSV must use UTF-8 encoding") from error
    return data


def _read_frame(name: str, data: bytes) -> pd.DataFrame:
    payload = _validated_bytes(name, data)
    try:
        frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"The CSV could not be parsed: {error}") from error
    if frame.empty:
        raise ValueError("The CSV has a header but no data rows")
    if not len(frame.columns) or any(not str(column).strip() for column in frame.columns):
        raise ValueError("Every CSV column needs a non-empty name")
    if len(set(map(str, frame.columns))) != len(frame.columns):
        raise ValueError("CSV column names must be unique")
    return frame


def inspect_custom_csv(name: str, data: bytes) -> CustomCSVInspection:
    """Read bounded metadata before the user maps timestamp and target columns."""

    frame = _read_frame(name, data)
    return CustomCSVInspection(
        columns=tuple(map(str, frame.columns)),
        row_count=len(frame),
        byte_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def infer_daily_seasonal_period(
    name: str,
    data: bytes,
    timestamp_column: str,
) -> int | None:
    """Return observations per 24 hours when the timestamps are regular."""

    frame = _read_frame(name, data)
    if timestamp_column not in frame:
        raise ValueError("The selected timestamp column is not present")
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    if timestamps.isna().any() or len(timestamps) < 2:
        return None
    diffs = timestamps.diff().dropna()
    if diffs.empty or diffs.nunique() != 1 or diffs.iloc[0] <= pd.Timedelta(0):
        return None
    interval_seconds = diffs.iloc[0].total_seconds()
    slots = 24 * 60 * 60 / interval_seconds
    rounded = round(slots)
    return int(rounded) if rounded >= 2 and math.isclose(slots, rounded) else None


def _numeric_series(frame: pd.DataFrame, column: str, role: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"The selected {role} column is not present")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        bad = int(values.isna().sum())
        raise ValueError(f"{role.capitalize()} column '{column}' has {bad} blank or non-numeric value(s)")
    if not values.map(math.isfinite).all():
        raise ValueError(f"{role.capitalize()} column '{column}' contains a non-finite value")
    return values.astype(float)


def prepare_custom_csv(
    name: str,
    data: bytes,
    timestamp_column: str,
    target_column: str,
    covariate_columns: list[str] | tuple[str, ...] = (),
    seasonal_period: int = 48,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    unit: str = "energy units",
) -> tuple[dict, dict]:
    """Map, validate, feature-engineer and chronologically split a custom CSV.

    Covariates must be numeric and known at the forecast target time. The
    caller is responsible for presenting that restriction to the user.
    """

    if timestamp_column == target_column:
        raise ValueError("Timestamp and target must use different columns")
    covariates = tuple(dict.fromkeys(covariate_columns))
    if timestamp_column in covariates or target_column in covariates:
        raise ValueError("Timestamp and target cannot also be selected as covariates")
    if not isinstance(seasonal_period, int) or seasonal_period < 2:
        raise ValueError("Seasonal period must be an integer of at least 2 rows")
    if not (0.5 <= train_fraction < 0.9 and 0.05 <= validation_fraction < 0.3):
        raise ValueError("Train and validation fractions are outside the supported range")
    if train_fraction + validation_fraction >= 0.95:
        raise ValueError("At least 5% of examples must remain for the reserved test set")

    frame = _read_frame(name, data)
    if timestamp_column not in frame:
        raise ValueError("The selected timestamp column is not present")
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError(
            f"Timestamp column '{timestamp_column}' has {int(timestamps.isna().sum())} invalid value(s)"
        )
    if timestamps.duplicated().any():
        raise ValueError("Timestamp column contains duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Rows must be in strictly increasing timestamp order")
    diffs = timestamps.diff().dropna()
    if diffs.empty or diffs.nunique() != 1 or diffs.iloc[0] <= pd.Timedelta(0):
        raise ValueError("Timestamps must have one regular positive sampling interval")

    target = _numeric_series(frame, target_column, "target")
    if (target < 0).any():
        raise ValueError("Energy target values must not be negative")
    covariate_values = {
        column: _numeric_series(frame, column, "covariate")
        for column in covariates
    }

    examples: list[dict] = []
    for index in range(seasonal_period, len(frame)):
        key = timestamps.iloc[index].to_pydatetime()
        history = tuple(float(value) for value in target.iloc[index - seasonal_period:index])
        examples.append({
            "key": key,
            "target": float(target.iloc[index]),
            "lag_1": float(target.iloc[index - 1]),
            # Existing model contracts use lag_48/history_48; for a custom
            # dataset these hold the configured seasonal lag/window.
            "lag_48": float(target.iloc[index - seasonal_period]),
            "seasonal_naive": float(target.iloc[index - seasonal_period]),
            "history_48": history,
            "target_slot": index % seasonal_period,
            "slots_per_season": seasonal_period,
            "custom_covariates": {
                column: float(values.iloc[index])
                for column, values in covariate_values.items()
            },
        })

    if len(examples) < 3 * MIN_PART_EXAMPLES:
        raise ValueError(
            "The CSV is too short after seasonal feature creation; at least "
            f"{seasonal_period + 3 * MIN_PART_EXAMPLES} rows are required"
        )
    train_end = int(len(examples) * train_fraction)
    validation_end = int(len(examples) * (train_fraction + validation_fraction))
    parts = {
        "train": examples[:train_end],
        "validation": examples[train_end:validation_end],
        "test": examples[validation_end:],
    }
    if any(len(part) < MIN_PART_EXAMPLES for part in parts.values()):
        raise ValueError("Each chronological partition needs at least 10 forecast targets")

    interval = diffs.iloc[0]
    profile = {
        "source_file": Path(name).name,
        "row_count": len(frame),
        "timestamp_column": timestamp_column,
        "target_column": target_column,
        "covariate_columns": list(covariates),
        "sampling_interval": str(interval),
        "seasonal_period_rows": seasonal_period,
        "forecast_horizon_rows": 1,
        "start": timestamps.iloc[0].isoformat(),
        "end": timestamps.iloc[-1].isoformat(),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
        "unit": unit.strip() or "energy units",
        "missing_values": 0,
        "duplicate_timestamps": 0,
    }
    boundaries = {
        "validation_start": parts["validation"][0]["key"],
        "test_start": parts["test"][0]["key"],
    }
    evidence = {
        "custom_uploads": [{
            "safe_name": Path(name).name,
            "byte_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }],
        "accepted_upload_count": 1,
        "schema_mapping": {
            "timestamp": timestamp_column,
            "target": target_column,
            "known_covariates": list(covariates),
        },
    }
    return {
        "label": f"Custom CSV - {target_column}",
        "unit": profile["unit"],
        "profile": profile,
        "boundaries": boundaries,
        "parts": parts,
    }, evidence
