"""Presentation helpers shared by the Streamlit dashboard and its tests."""

from __future__ import annotations

from datetime import datetime, timedelta
import re


_AUSGRID_TARGET = re.compile(r"^(\d{4}-\d{2}-\d{2}) slot (\d+)$")


def forecast_plot_timestamp(case: str, target_key: str) -> datetime:
    """Convert a serialised validation target into a plottable timestamp.

    AEMO targets are already ISO timestamps. Ausgrid targets use the documented
    ``YYYY-MM-DD slot NN`` label, where each slot represents 30 minutes.
    """

    if case in {"aemo", "custom"}:
        return datetime.fromisoformat(target_key)

    if case == "ausgrid":
        match = _AUSGRID_TARGET.fullmatch(target_key)
        if match is None:
            raise ValueError(f"Invalid Ausgrid target key: {target_key!r}")
        source_day = datetime.fromisoformat(match.group(1))
        source_slot = int(match.group(2))
        return source_day + timedelta(minutes=30 * source_slot)

    raise ValueError(f"Unsupported forecasting case: {case!r}")
