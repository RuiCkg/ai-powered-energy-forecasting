"""Past-only 48-value input windows for the LSTM; no target or future value is included."""

from __future__ import annotations

WINDOW = 48


def aemo_history_windows(records: list[dict], examples: list[dict]) -> list[dict]:
    """Attach the 48 half-hour actuals that precede each AEMO target.

    ``history_48`` ends at the forecast origin (target - 30 min), so the target
    itself is never part of its own input.
    """

    index_by_time = {record["timestamp"]: index for index, record in enumerate(records)}
    windowed = []
    for example in examples:
        end = index_by_time[example["key"]]
        if end < WINDOW:
            raise ValueError("AEMO example lacks 48 past values")
        history = tuple(float(record["target"]) for record in records[end - WINDOW:end])
        windowed.append({**example, "history_48": history})
    return windowed


def ausgrid_history_windows(rows: list[dict], examples: list[dict]) -> list[dict]:
    """Attach the 48 source-order PV values that precede each Ausgrid target.

    Days are concatenated in source order (no daylight-saving timestamps are
    invented). ``example["key"]`` is ``(date, 1-based slot)``.
    """

    flat = [float(value) for row in rows for value in row["values_kWh"]]
    day_index = {row["date"]: index for index, row in enumerate(rows)}
    windowed = []
    for example in examples:
        day, slot = example["key"]
        end = day_index[day] * 48 + (slot - 1)
        if end < WINDOW:
            raise ValueError("Ausgrid example lacks 48 past values")
        if flat[end] != float(example["target"]):
            raise ValueError("Ausgrid window is not aligned with its target")
        windowed.append({**example, "history_48": tuple(flat[end - WINDOW:end])})
    return windowed
