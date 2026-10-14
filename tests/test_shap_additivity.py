"""Cover the SHAP additivity check, which guards an attribution claim.

The check existed before this workstream but compared a float32 sum against the
prediction with a fixed absolute tolerance, and only for the first validation
row. On the load case, whose predictions are in the thousands of MW, float32
cannot represent a difference as small as that tolerance, so well-formed data
failed it roughly seven rows in ten — and whether it failed at all depended on
which row happened to be first.

These tests pin the replacement: the error is scaled by the prediction, it is
taken across every row, and a genuine additivity break is still caught. They use
hand-built arrays, so no model is trained and xgboost is not needed.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.xgb_model import SHAP_ADDITIVITY_TOLERANCE, shap_additivity_error


class ShapAdditivityErrorTest(unittest.TestCase):
    def test_exact_contributions_have_no_error(self):
        contributions = [[1.0, 2.0, 3.0, 6500.0], [2.0, 2.0, 2.0, 7000.0]]
        self.assertEqual(shap_additivity_error(contributions, [6506.0, 7006.0]), 0.0)

    def test_float32_scale_noise_on_a_large_prediction_passes(self):
        """The case the old absolute tolerance failed: ~8e-3 on ~6,500 MW."""

        raw = 6500.0
        contributions = [[1000.0, 2000.0, 3000.0, 500.0 + 0.0078]]
        error = shap_additivity_error(contributions, [raw])
        self.assertLess(error, SHAP_ADDITIVITY_TOLERANCE)
        # The same discrepancy measured absolutely exceeds the old 1e-3 threshold,
        # which is exactly why that threshold was wrong at this magnitude.
        self.assertGreater(0.0078, 1e-3)

    def test_a_real_additivity_break_is_caught(self):
        """A whole-prediction error is orders of magnitude above the tolerance."""

        contributions = [[1000.0, 2000.0, 3000.0, 400.0]]
        self.assertGreater(shap_additivity_error(contributions, [6500.0]), SHAP_ADDITIVITY_TOLERANCE)

    def test_near_zero_predictions_keep_an_absolute_tolerance(self):
        """The PV case predicts near zero, so the scale must not divide by it."""

        contributions = [[0.1, 0.2, 0.0, 0.0000001]]
        error = shap_additivity_error(contributions, [0.3])
        self.assertLess(error, SHAP_ADDITIVITY_TOLERANCE)
        broken = shap_additivity_error([[0.1, 0.2, 0.0, 0.5]], [0.3])
        self.assertGreater(broken, SHAP_ADDITIVITY_TOLERANCE)

    def test_the_worst_row_decides_not_the_first(self):
        """The old check looked at row 0 only, which made it a lottery."""

        good_first = [[1000.0, 2000.0, 3000.0, 500.0], [1000.0, 2000.0, 3000.0, 400.0]]
        self.assertGreater(shap_additivity_error(good_first, [6500.0, 6500.0]), SHAP_ADDITIVITY_TOLERANCE)

    def test_misaligned_or_empty_input_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not aligned"):
            shap_additivity_error([[1.0, 2.0]], [3.0, 4.0])
        with self.assertRaisesRegex(ValueError, "No predictions"):
            shap_additivity_error([], [])


if __name__ == "__main__":
    unittest.main()
