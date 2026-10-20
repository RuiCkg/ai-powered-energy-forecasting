"""Deterministic synthetic data used only for the guided product demonstration."""

from __future__ import annotations

from datetime import timedelta
import math

import pandas as pd

from .custom_data import MIN_PART_EXAMPLES


def build_synthetic_demo_csv(row_count: int = 672) -> bytes:
    """Return a valid two-week half-hourly demonstration CSV."""

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
