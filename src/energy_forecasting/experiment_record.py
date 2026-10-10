"""Record what a run was, so the same configuration can be replayed and checked.

NFR5 is not satisfied by a run that can be repeated approximately. The test is
that a recorded configuration is replayed and the reported figures are the
*same*, not close, so the record has to contain everything that moves a
number: library versions, seeds, the fixed model configuration, the feature
list, the chronological split boundaries, and a digest of the exact source
bytes that were accepted. A record missing any of those cannot fail the
comparison honestly, because a mismatch could always be blamed on something
unrecorded (Pineau et al., 2021).

The record is aggregate-only by construction. ``assert_record_value_free``
refuses a record carrying per-row source readings or predictions, which is how
NFR4 is checked in code rather than by reviewing a file after the fact.

Comparison is exact. ``canonical_json`` fixes key order and float formatting so
two records of the same run produce byte-identical text and the same digest.
"""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import platform
import sys

from .upload_validation import AEMO_MONTHLY_POLICY, AUSGRID_ARCHIVE_POLICY, MAX_SESSION_BYTES
from .xgb_model import EARLY_STOPPING_ROUNDS, FEATURE_NAMES, FIXED_PARAMETERS, NUM_BOOST_ROUND

RECORD_SCHEMA_VERSION = "1"

# Packages whose version can change a reported metric. Absent packages are
# recorded as null rather than omitted, so a record made without the LSTM
# extras is still comparable with one made with them.
RECORDED_PACKAGES = ("numpy", "scipy", "xgboost-cpu", "xgboost", "torch", "streamlit")

# Containers that hold per-row source readings or predictions. A record with
# any of these has leaked dataset values into an artefact meant to be shareable
# evidence (NFR4).
VALUE_BEARING_KEYS = frozenset({
    "validation_prediction_rows",
    "validation_first_local_SHAP",
    "values_kWh",
    "history_48",
    "clock_labels",
    "records",
    "rows",
    "examples",
    "predictions",
})

# Names that mean a reading when they hold a series, or when they sit inside a
# list element, and mean something else entirely when they are a single
# aggregate keyed by feature. ``lag_1`` is both a reading and a feature name:
# ``{"lag_1": [1.0, 2.0]}`` is a series of readings, while
# ``{"validation_mean_absolute_SHAP_by_feature": {"lag_1": 120.5}}`` is one
# summary statistic. Only the first is a leak.
PER_ROW_VALUE_KEYS = frozenset({
    "target",
    "lag_1",
    "lag_48",
    "seasonal_naive",
    "actual",
    "predicted",
    "source_clock_label",
})

# The preprocessing decisions in force. These are statements about the current
# code, not aspirations: each one is enforced by a parser, a split function or
# a model wrapper, and any change to them changes the record.
PREPROCESSING_POLICY = {
    "imputation": "none; a gap or blank interval fails source validation instead of being filled",
    "negative_or_non_finite_readings": "rejected at parse time, not clipped",
    "outlier_handling": "none applied; anomalous readings are flagged in the data manifest and retained",
    "timestamp_policy": "AEMO fixed NEM UTC+10; Ausgrid kept as source date plus slot, daylight saving unresolved",
    "split_policy": "chronological only; boundaries recorded and enforced in code, never a random split",
    "prediction_postprocessing": "predictions clipped at zero; clipping happens after SHAP attribution",
    "test_period": "reserved; not used for fitting, normalisation, selection or reported metrics",
}


def _package_versions() -> dict:
    from importlib import metadata

    versions = {}
    for name in RECORDED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_fingerprint() -> dict:
    """Describe the interpreter and package versions a replay has to match."""

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "maxsize_64_bit": sys.maxsize > 2**32,
        "package_versions": _package_versions(),
    }


def fixed_model_configuration() -> dict:
    """The seeds and hyperparameters that are fixed in code, read from the code.

    Read from the model modules rather than restated here, so a record cannot
    quietly disagree with the run that produced it.
    """

    from . import lstm_model

    return {
        "xgboost": {
            "fixed_parameters": dict(FIXED_PARAMETERS),
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "feature_names": list(FEATURE_NAMES),
            "selection": "tree count chosen on validation MAE",
        },
        "lstm": {
            "fixed_configuration": dict(lstm_model.FIXED_CONFIG),
            "selection": "epoch checkpoint chosen on validation MAE",
        },
    }


def _serialisable(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _serialisable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialisable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    return str(value)


def canonical_json(value) -> str:
    """Serialise deterministically so equal runs give byte-identical text.

    Sorted keys and no separator padding remove the two ways an equivalent
    record can differ textually. Floats use Python's shortest round-trip
    representation, so an exact metric comparison stays exact.
    """

    return json.dumps(_serialisable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value) -> str:
    """Return the SHA-256 digest of the canonical serialisation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _find_value_bearing_keys(node, path: str = "", inside_list: bool = False) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, item in node.items():
            where = f"{path}.{key}" if path else str(key)
            if key in VALUE_BEARING_KEYS:
                found.append(where)
            elif key in PER_ROW_VALUE_KEYS and (inside_list or isinstance(item, (list, tuple))):
                found.append(where)
            found.extend(_find_value_bearing_keys(item, where, inside_list))
    elif isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            found.extend(_find_value_bearing_keys(item, f"{path}[{index}]", inside_list=True))
    return found


def assert_record_value_free(record: dict) -> None:
    """Raise if a record carries per-row source readings or predictions (NFR4)."""

    offending = _find_value_bearing_keys(record)
    if offending:
        raise ValueError(f"Experiment record carries dataset values at: {sorted(offending)}")


def aggregate_model_result(result: dict) -> dict:
    """Keep the parts of a model result that are aggregate evidence.

    Per-row previews and the single local SHAP explanation are dropped. They
    are useful in the interface and they are source values, so they belong in a
    download the user chose, not in the shareable record.
    """

    keep = (
        "model",
        "xgboost_version",
        "torch_version",
        "fixed_parameters",
        "best_boosting_iteration_selected_on_validation",
        "best_epoch_selected_on_validation",
        "epochs_run",
        "hidden_size",
        "optimizer",
        "max_epochs",
        "early_stop_patience_epochs",
        "feature_names",
        "nonnegative_predictions",
        "validation_metrics",
        "validation_mean_absolute_SHAP_by_feature",
        "validation_seasonal_naive_metrics",
        "validation_persistence_metrics",
        "SHAP_scope",
        "test_status",
    )
    return {key: result[key] for key in keep if key in result}


def build_experiment_record(
    case: str,
    upload_evidence: dict,
    source_profile: dict,
    split_boundaries: dict,
    part_counts: dict,
    model_results: dict,
) -> dict:
    """Assemble a replayable record for one case, and check it for source values.

    ``configuration`` holds everything a replay must reproduce and is digested
    on its own, so two runs can be shown to share a configuration before their
    figures are compared. ``metrics`` is digested separately; NFR5 is met when
    identical configuration digests produce identical metric digests.
    """

    configuration = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "case": case,
        "environment": environment_fingerprint(),
        "fixed_model_configuration": fixed_model_configuration(),
        "preprocessing_policy": dict(PREPROCESSING_POLICY),
        "upload_policies": [AEMO_MONTHLY_POLICY.describe(), AUSGRID_ARCHIVE_POLICY.describe()],
        "session_byte_budget": MAX_SESSION_BYTES,
        "accepted_source_digests": {
            "aemo": [item["sha256"] for item in upload_evidence.get("aemo_uploads", [])],
            "ausgrid": [item["sha256"] for item in upload_evidence.get("ausgrid_uploads", [])],
        },
        "source_profile": _serialisable(source_profile),
        "split_boundaries": _serialisable(split_boundaries),
        "example_counts": _serialisable(part_counts),
    }
    # Serialised on the way in, which also detaches the record from the caller's
    # dictionaries. A record that still shares structure with a live result is
    # not a snapshot, and evidence that can change after the fact is not
    # evidence.
    metrics = {name: _serialisable(aggregate_model_result(result)) for name, result in sorted(model_results.items())}
    record = {
        "configuration": configuration,
        "configuration_digest": digest(configuration),
        "metrics": metrics,
        "metrics_digest": digest(metrics),
        "replay_criterion": (
            "NFR5 is met when a replay of this configuration reproduces metrics_digest exactly. "
            "Equal configuration_digest with unequal metrics_digest is a failure, not a tolerance question."
        ),
    }
    assert_record_value_free(record)
    return record


def compare_records(first: dict, second: dict) -> dict:
    """Compare two records for exact replay, reporting where they diverge.

    A same-configuration, same-metrics pair passes. Anything else returns the
    differing paths, because "it did not reproduce" is only actionable with the
    field that moved.
    """

    configuration_matches = first.get("configuration_digest") == second.get("configuration_digest")
    metrics_match = first.get("metrics_digest") == second.get("metrics_digest")
    differences: list[str] = []
    if not configuration_matches:
        differences.extend(_diff_paths(first.get("configuration"), second.get("configuration"), "configuration"))
    if not metrics_match:
        differences.extend(_diff_paths(first.get("metrics"), second.get("metrics"), "metrics"))
    return {
        "configuration_matches": configuration_matches,
        "metrics_match": metrics_match,
        "replay_reproduced": bool(configuration_matches and metrics_match),
        "differing_paths": sorted(differences),
    }


def _diff_paths(first, second, path: str) -> list[str]:
    if isinstance(first, dict) and isinstance(second, dict):
        differences: list[str] = []
        for key in sorted(set(first) | set(second)):
            where = f"{path}.{key}"
            if key not in first or key not in second:
                differences.append(where)
            else:
                differences.extend(_diff_paths(first[key], second[key], where))
        return differences
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return [f"{path}[length]"]
        return [
            difference
            for index, (left, right) in enumerate(zip(first, second))
            for difference in _diff_paths(left, right, f"{path}[{index}]")
        ]
    return [] if canonical_json(first) == canonical_json(second) else [path]
