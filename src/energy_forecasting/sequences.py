"""Attach 48 past observed source values to the existing one-step examples."""

from __future__ import annotations


def aemo_history_windows(records: list[dict], examples: list[dict]) -> list[dict]:
    """AEMO 48-value history ending at the immediately completed interval."""

    if len(examples) != max(0, len(records) - 48):
        raise ValueError("AEMO examples do not match the source record coverage")
    result = []
    for index, example in enumerate(examples, start=48):
        if example["key"] != records[index]["timestamp"]:
            raise ValueError("AEMO sequence target key is not aligned")
        history = tuple(record["target"] for record in records[index - 48:index])
        if history[0] != example["lag_48"] or history[-1] != example["lag_1"]:
            raise ValueError("AEMO history contains an unexpected lag")
        result.append({**example, "history_48": history})
    return result


def ausgrid_history_windows(rows: list[dict], examples: list[dict]) -> list[dict]:
    """Use 48 previous source slots, never assign an absolute DST timestamp."""

    expected = max(0, len(rows) - 1) * 47
    if len(examples) != expected:
        raise ValueError("Ausgrid examples do not match selected source days")
    values = [value for row in rows for value in row["values_kWh"]]
    result = []
    cursor = 0
    for day_index in range(1, len(rows)):
        for slot_index in range(1, 48):
            example = examples[cursor]
            cursor += 1
            if example["key"] != (rows[day_index]["date"], slot_index + 1):
                raise ValueError("Ausgrid sequence target key is not aligned")
            target_index = day_index * 48 + slot_index
            history = tuple(values[target_index - 48:target_index])
            if len(history) != 48 or history[0] != example["lag_48"] or history[-1] != example["lag_1"]:
                raise ValueError("Ausgrid history contains an unexpected lag")
            result.append({**example, "history_48": history})
    return result
