"""Guard the one ordering invariant the interface must not lose.

Validation is only worth anything if it runs before a parser touches the bytes.
That ordering lives in ``streamlit_app.py``, where a later edit could reverse it
without any module-level test noticing, so it is asserted here. The checks are
deliberately text-based and import nothing, so they run in the documented test
command whether or not the app extras are installed.
"""

from pathlib import Path
import unittest

APP_SOURCE = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8")


class AppWiringTest(unittest.TestCase):
    def test_uploads_are_validated_before_they_are_parsed(self):
        validate_at = APP_SOURCE.find("validate_documented_sources(aemo_files, ausgrid_file)")
        parse_at = APP_SOURCE.find("prepare_uploaded_sources(aemo_files, ausgrid_file)")
        self.assertNotEqual(validate_at, -1, "the app must call the upload validation gate")
        self.assertNotEqual(parse_at, -1, "the app must still call the source parser")
        self.assertLess(validate_at, parse_at, "validation must run before any parser sees the bytes")

    def test_rejections_are_reported_by_code_and_fixed_message(self):
        self.assertIn("except UploadRejected as rejection:", APP_SOURCE)
        self.assertIn("rejection.code", APP_SOURCE)
        self.assertIn("rejection.message", APP_SOURCE)

    def test_a_rejection_clears_any_previously_prepared_state(self):
        """A refused upload must not leave an earlier accepted run on screen."""

        start = APP_SOURCE.find("except UploadRejected")
        end = APP_SOURCE.find("except (ValueError", start)
        block = APP_SOURCE[start:end]
        self.assertIn('st.session_state.pop("prepared", None)', block)
        self.assertIn('st.session_state.pop("upload_evidence", None)', block)

    def test_the_experiment_record_is_built_from_the_recorded_boundaries(self):
        self.assertIn("build_experiment_record(", APP_SOURCE)
        self.assertIn('case_data["boundaries"]', APP_SOURCE)
        self.assertIn('st.session_state.get("upload_evidence", {})', APP_SOURCE)

    def test_the_file_picker_type_filter_is_not_the_only_control(self):
        """``type="zip"`` is a convenience; the server-side rule must exist too."""

        self.assertIn('type="zip"', APP_SOURCE)
        self.assertIn("validate_documented_sources", APP_SOURCE)

    def test_the_user_journey_has_progress_and_result_sections(self):
        self.assertIn('class="workflow-steps"', APP_SOURCE)
        for mode in ("Try demo data", "Upload your CSV", "Official benchmark · Advanced"):
            self.assertIn(f'"{mode}"', APP_SOURCE)
        for label in ("Overview", "Forecasts", "Explainability (XGBoost)", "Downloads"):
            self.assertIn(f'"{label}"', APP_SOURCE)

    def test_model_specific_results_are_not_presented_as_last_run_results(self):
        self.assertIn('first.metric("Best model by MAE"', APP_SOURCE)
        self.assertIn("not the model that was run most recently", APP_SOURCE)
        self.assertIn("This tab explains the XGBoost model only", APP_SOURCE)

    def test_the_forecast_chart_does_not_rely_on_colour_alone(self):
        self.assertIn("strokeDash=alt.StrokeDash", APP_SOURCE)
        self.assertIn("actual (solid) and predicted (dashed)", APP_SOURCE)

    def test_the_visible_tables_are_readable_and_use_current_width_api(self):
        self.assertIn('.round(metric_decimals)', APP_SOURCE)
        self.assertIn('width="stretch"', APP_SOURCE)
        self.assertNotIn("use_container_width", APP_SOURCE)

    def test_model_completion_refreshes_status_without_duplicate_button_rows(self):
        self.assertIn('st.session_state["completion_notice"]', APP_SOURCE)
        self.assertIn("on_click=run_validation_model", APP_SOURCE)
        self.assertEqual(APP_SOURCE.count('key=f"xgb_{case}"'), 1)
        self.assertEqual(APP_SOURCE.count('key=f"lstm_{case}"'), 1)
        self.assertNotIn("st.rerun()", APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
