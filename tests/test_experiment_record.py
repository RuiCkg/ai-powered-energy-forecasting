"""Evidence for NFR5 and for NFR4's no-values rule in the record itself.

The replay criterion is exact, so these tests check exactness rather than
closeness: a record built twice from the same inputs must give the same
digests, a single changed metric must fail the comparison and name the field,
and a record carrying per-row readings must be refused outright.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting import lstm_model
from energy_forecasting.experiment_record import (
    RECORD_SCHEMA_VERSION,
    aggregate_model_result,
    assert_record_value_free,
    build_experiment_record,
    canonical_json,
    compare_records,
    digest,
    environment_fingerprint,
    fixed_model_configuration,
)
from energy_forecasting.xgb_model import FEATURE_NAMES, FIXED_PARAMETERS

NEM = timezone(timedelta(hours=10))

UPLOAD_EVIDENCE = {
    "aemo_uploads": [{"sha256": "a" * 64, "byte_size": 1024}, {"sha256": "b" * 64, "byte_size": 2048}],
    "ausgrid_uploads": [{"sha256": "c" * 64, "byte_size": 4096}],
}
SOURCE_PROFILE = {"region": "NSW1", "unit": "MW", "count": 17520, "non_30_minute_gaps": 0}
SPLIT_BOUNDARIES = {
    "validation_start": datetime(2026, 4, 1, 4, 30, tzinfo=NEM),
    "test_start": datetime(2026, 6, 1, 4, 30, tzinfo=NEM),
}
PART_COUNTS = {"train": 11000, "validation": 2900, "test": 2900}
MODEL_RESULTS = {
    "XGBoost": {
        "model": "XGBoost CPU, squared-error trees",
        "fixed_parameters": dict(FIXED_PARAMETERS),
        "feature_names": list(FEATURE_NAMES),
        "best_boosting_iteration_selected_on_validation": 217,
        "validation_metrics": {"MAE": 91.3125, "RMSE": 128.5, "positive_actual_MAPE": 1.0625},
        "validation_mean_absolute_SHAP_by_feature": {"lag_1": 120.5, "lag_48": 42.25},
        "validation_prediction_rows": [{"key": "2026-04-01T05:00:00+10:00", "actual": 7421.5, "predicted": 7390.25}],
        "validation_first_local_SHAP": {"bias": 8000.0, "raw_prediction_before_clipping": 7390.25},
        "test_status": "reserved",
    }
}


def a_record():
    return build_experiment_record(
        "aemo", UPLOAD_EVIDENCE, SOURCE_PROFILE, SPLIT_BOUNDARIES, PART_COUNTS, MODEL_RESULTS
    )


class CanonicalSerialisationTest(unittest.TestCase):
    def test_key_order_does_not_change_the_digest(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))

    def test_datetimes_serialise_as_recorded_boundaries(self):
        self.assertIn("2026-04-01T04:30:00+10:00", canonical_json(SPLIT_BOUNDARIES))

    def test_floats_keep_an_exact_round_trip_representation(self):
        self.assertIn("91.3125", canonical_json({"MAE": 91.3125}))
        self.assertNotEqual(digest({"MAE": 91.3125}), digest({"MAE": 91.3126}))

    def test_non_finite_metrics_are_refused_rather_than_written_as_nan(self):
        with self.assertRaises(ValueError):
            canonical_json({"MAE": float("nan")})


class RecordContentTest(unittest.TestCase):
    def test_record_captures_what_a_replay_needs(self):
        configuration = a_record()["configuration"]
        self.assertEqual(configuration["schema_version"], RECORD_SCHEMA_VERSION)
        self.assertEqual(configuration["case"], "aemo")
        self.assertEqual(configuration["accepted_source_digests"]["aemo"], ["a" * 64, "b" * 64])
        self.assertEqual(configuration["example_counts"], PART_COUNTS)
        self.assertIn("validation_start", configuration["split_boundaries"])
        self.assertIn("python_version", configuration["environment"])
        self.assertIn("package_versions", configuration["environment"])
        self.assertEqual(configuration["preprocessing_policy"]["imputation"][:4], "none")
        self.assertIn("chronological only", configuration["preprocessing_policy"]["split_policy"])

    def test_fixed_configuration_is_read_from_the_model_code(self):
        """A restated copy could drift from the run; this reads the real constants."""

        fixed = fixed_model_configuration()
        self.assertEqual(fixed["xgboost"]["fixed_parameters"]["seed"], FIXED_PARAMETERS["seed"])
        self.assertEqual(fixed["xgboost"]["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(fixed["lstm"]["fixed_configuration"]["seed"], lstm_model.FIXED_CONFIG["seed"])
        self.assertEqual(
            fixed["lstm"]["fixed_configuration"]["max_epochs"], lstm_model.FIXED_CONFIG["max_epochs"]
        )

    def test_environment_records_absent_packages_as_null_not_missing(self):
        versions = environment_fingerprint()["package_versions"]
        for name in ("numpy", "torch", "streamlit"):
            self.assertIn(name, versions)

    def test_per_row_values_are_dropped_from_the_record(self):
        """NFR4: the record is aggregate evidence, so previews do not enter it."""

        metrics = a_record()["metrics"]["XGBoost"]
        self.assertNotIn("validation_prediction_rows", metrics)
        self.assertNotIn("validation_first_local_SHAP", metrics)
        self.assertEqual(metrics["validation_metrics"]["MAE"], 91.3125)
        self.assertIn("validation_mean_absolute_SHAP_by_feature", metrics)

    def test_aggregate_model_result_keeps_only_known_keys(self):
        aggregated = aggregate_model_result(MODEL_RESULTS["XGBoost"])
        self.assertNotIn("validation_prediction_rows", aggregated)
        self.assertIn("best_boosting_iteration_selected_on_validation", aggregated)

    def test_a_record_carrying_dataset_values_is_refused(self):
        polluted = a_record()
        polluted["metrics"]["XGBoost"]["validation_prediction_rows"] = [{"actual": 7421.5}]
        with self.assertRaisesRegex(ValueError, "carries dataset values"):
            assert_record_value_free(polluted)

    def test_building_a_record_with_planted_rows_fails_at_build_time(self):
        with self.assertRaisesRegex(ValueError, "carries dataset values"):
            build_experiment_record(
                "aemo", UPLOAD_EVIDENCE, {"rows": [{"target": 1.0}]}, SPLIT_BOUNDARIES, PART_COUNTS, MODEL_RESULTS
            )


class ReplayComparisonTest(unittest.TestCase):
    def test_same_inputs_reproduce_the_same_digests(self):
        first, second = a_record(), a_record()
        self.assertEqual(first["configuration_digest"], second["configuration_digest"])
        self.assertEqual(first["metrics_digest"], second["metrics_digest"])
        self.assertTrue(compare_records(first, second)["replay_reproduced"])

    def test_a_metric_that_moves_fails_the_replay_and_is_named(self):
        first, second = a_record(), a_record()
        second["metrics"]["XGBoost"]["validation_metrics"]["MAE"] = 91.3126
        second["metrics_digest"] = digest(second["metrics"])
        outcome = compare_records(first, second)
        self.assertTrue(outcome["configuration_matches"])
        self.assertFalse(outcome["metrics_match"])
        self.assertFalse(outcome["replay_reproduced"])
        self.assertIn("metrics.XGBoost.validation_metrics.MAE", outcome["differing_paths"])

    def test_a_close_metric_is_still_a_failure(self):
        """'Similar' does not satisfy the criterion; only equality does."""

        first, second = a_record(), a_record()
        second["metrics"]["XGBoost"]["validation_metrics"]["MAE"] = 91.31250000001
        second["metrics_digest"] = digest(second["metrics"])
        self.assertFalse(compare_records(first, second)["replay_reproduced"])

    def test_a_changed_source_digest_fails_the_configuration_comparison(self):
        first = a_record()
        second = build_experiment_record(
            "aemo",
            {"aemo_uploads": [{"sha256": "a" * 64}, {"sha256": "d" * 64}], "ausgrid_uploads": [{"sha256": "c" * 64}]},
            SOURCE_PROFILE, SPLIT_BOUNDARIES, PART_COUNTS, MODEL_RESULTS,
        )
        outcome = compare_records(first, second)
        self.assertFalse(outcome["configuration_matches"])
        self.assertIn("configuration.accepted_source_digests.aemo[1]", outcome["differing_paths"])

    def test_a_changed_split_boundary_fails_the_configuration_comparison(self):
        first = a_record()
        second = build_experiment_record(
            "aemo", UPLOAD_EVIDENCE, SOURCE_PROFILE,
            {"validation_start": datetime(2026, 3, 1, 4, 30, tzinfo=NEM), "test_start": SPLIT_BOUNDARIES["test_start"]},
            PART_COUNTS, MODEL_RESULTS,
        )
        self.assertIn("configuration.split_boundaries.validation_start", compare_records(first, second)["differing_paths"])

    def test_a_differing_seed_would_fail_the_comparison(self):
        """The seed is in the record, so a silently changed seed cannot pass."""

        first = a_record()
        second = a_record()
        second["configuration"]["fixed_model_configuration"]["xgboost"]["fixed_parameters"]["seed"] = 7
        second["configuration_digest"] = digest(second["configuration"])
        outcome = compare_records(first, second)
        self.assertFalse(outcome["configuration_matches"])
        self.assertIn(
            "configuration.fixed_model_configuration.xgboost.fixed_parameters.seed", outcome["differing_paths"]
        )

    def test_criterion_is_stated_in_the_record(self):
        self.assertIn("exactly", a_record()["replay_criterion"])

    def test_the_record_is_a_snapshot_not_a_live_view(self):
        """Evidence that changes when the caller's results change is not evidence."""

        results = {"XGBoost": {**MODEL_RESULTS["XGBoost"], "validation_metrics": {"MAE": 91.3125}}}
        record = build_experiment_record(
            "aemo", UPLOAD_EVIDENCE, SOURCE_PROFILE, SPLIT_BOUNDARIES, PART_COUNTS, results
        )
        results["XGBoost"]["validation_metrics"]["MAE"] = 0.0
        self.assertEqual(record["metrics"]["XGBoost"]["validation_metrics"]["MAE"], 91.3125)
        self.assertEqual(record["metrics_digest"], digest(record["metrics"]))

    def test_a_per_row_reading_under_any_key_is_caught(self):
        with self.assertRaisesRegex(ValueError, "carries dataset values"):
            assert_record_value_free({"anything": {"sample": [{"target": 7421.5}]}})
        with self.assertRaisesRegex(ValueError, "carries dataset values"):
            assert_record_value_free({"anything": {"lag_1": [1.0, 2.0]}})
        # A feature-keyed aggregate is not a reading and must stay allowed.
        assert_record_value_free({"validation_mean_absolute_SHAP_by_feature": {"lag_1": 120.5}})


if __name__ == "__main__":
    unittest.main()
