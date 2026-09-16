"""Causal one-step examples and previous-day seasonal-naive evaluation."""

from __future__ import annotations

from datetime import timedelta
from math import sqrt


def aemo_one_step_examples(records: list[dict]) -> list[dict]:
    """Target at t, origin at t-30 min; previous-day value is the baseline."""

    examples = []
    for index in range(48, len(records)):
        target = records[index]
        previous = records[index - 1]
        previous_day = records[index - 48]
        if target["timestamp"] - previous["timestamp"] != timedelta(minutes=30):
            raise ValueError("AEMO one-step history is not half-hourly")
        if target["timestamp"] - previous_day["timestamp"] != timedelta(days=1):
            raise ValueError("AEMO previous-day source history is not aligned")
        examples.append({
            "key": target["timestamp"],
            "target": target["target"],
            "lag_1": previous["target"],
            "lag_48": previous_day["target"],
            "seasonal_naive": previous_day["target"],
        })
    return examples


def ausgrid_next_slot_examples(rows: list[dict]) -> list[dict]:
    """Predict the next source slot within a day, not an absolute DST timestamp.

    The origin is the previous source slot of that day. The previous-day same
    slot is a causal seasonal-naive prediction. Day-boundary transitions are
    intentionally omitted pending a verified daylight-saving interpretation.
    """

    examples = []
    for index in range(1, len(rows)):
        current = rows[index]
        previous_day = rows[index - 1]
        if (current["date"] - previous_day["date"]).days != 1:
            raise ValueError("Ausgrid selected source days are not consecutive")
        for row in (current, previous_day):
            if row["row_quality"] is None or row["row_quality"] == "NA" or any(value is None for value in row["values_kWh"]):
                raise ValueError("Ausgrid source history fails the actual/no-blank selection rule")
        for slot in range(1, 48):
            examples.append({
                "key": (current["date"], slot + 1),
                "target": current["values_kWh"][slot],
                "lag_1": current["values_kWh"][slot - 1],
                "lag_48": previous_day["values_kWh"][slot],
                "seasonal_naive": previous_day["values_kWh"][slot],
                "source_clock_label": current["clock_labels"][slot],
            })
    return examples


def evaluate_seasonal_naive(examples: list[dict]) -> dict:
    """Score all valid targets with MAE/RMSE; MAPE only on positive actuals."""

    return evaluate_predictions(examples, [example["seasonal_naive"] for example in examples])


def evaluate_persistence(examples: list[dict]) -> dict:
    """Score the immediately preceding actual as a one-step forecast."""

    return evaluate_predictions(examples, [example["lag_1"] for example in examples])


def evaluate_predictions(examples: list[dict], predictions: list[float]) -> dict:
    """Score aligned forecasts; never include zero actuals in ordinary MAPE."""

    if not examples:
        raise ValueError("No forecast examples to evaluate")
    if len(examples) != len(predictions):
        raise ValueError("Prediction count does not match target count")
    errors = [prediction - example["target"] for example, prediction in zip(examples, predictions)]
    positive = [(example, prediction) for example, prediction in zip(examples, predictions) if example["target"] > 0]
    if any(example["target"] < 0 for example in examples):
        raise ValueError("Negative actual target in forecast evaluation")
    return {
        "example_count": len(examples),
        "MAE": sum(abs(error) for error in errors) / len(errors),
        "RMSE": sqrt(sum(error * error for error in errors) / len(errors)),
        "MAPE_positive_actual_percent": (
            100 * sum(abs(prediction - example["target"]) / example["target"] for example, prediction in positive) / len(positive)
            if positive else None
        ),
        "positive_actual_count_for_MAPE": len(positive),
        "zero_actual_count_excluded_from_MAPE": len(examples) - len(positive),
    }


def partition_examples(examples: list[dict], validation_start, test_start, source_date_key: bool = False) -> dict[str, list[dict]]:
    """Partition by forecast *target* key, preserving past history as input."""

    if not examples:
        raise ValueError("No forecast examples to partition")
    keys = [example["key"] for example in examples]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("Forecast target keys must be unique and chronological")
    if not validation_start < test_start:
        raise ValueError("Forecast partition boundaries must be in order")
    parts = {"train": [], "validation": [], "test": []}
    for example in examples:
        key = example["key"][0] if source_date_key else example["key"]
        if key < validation_start:
            parts["train"].append(example)
        elif key < test_start:
            parts["validation"].append(example)
        else:
            parts["test"].append(example)
    if any(not part for part in parts.values()):
        raise ValueError("Each forecast partition needs target examples")
    return parts


def preview_predictions(examples: list[dict], predictions: list[float], limit: int = 100) -> list[dict]:
    """Small in-memory actual-versus-forecast sample for the local dashboard."""

    if len(examples) != len(predictions):
        raise ValueError("Prediction preview is not target-aligned")
    preview = []
    for example, prediction in zip(examples[:limit], predictions[:limit]):
        key = example["key"]
        if isinstance(key, tuple):
            label = f"{key[0].isoformat()} slot {key[1]:02d}"
        else:
            label = key.isoformat()
        preview.append({"target_key": label, "actual": float(example["target"]), "predicted": float(prediction)})
    return preview
