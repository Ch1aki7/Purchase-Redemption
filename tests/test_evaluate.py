import unittest

import pandas as pd

from src.evaluate import evaluate_prediction


class EvaluatePredictionTest(unittest.TestCase):
    def test_perfect_prediction_has_zero_risk(self):
        frame = pd.DataFrame({
            "purchase_true": [100.0, 200.0],
            "purchase_pred": [100.0, 200.0],
            "redeem_true": [80.0, 120.0],
            "redeem_pred": [80.0, 120.0],
        })
        metrics = evaluate_prediction(frame)
        self.assertEqual(metrics["risk_adjusted_loss"], 0.0)
        self.assertEqual(metrics["purchase_over_30_days"], 0)
        self.assertEqual(metrics["total_score"], 10.0)

    def test_tail_and_severe_days_are_reported(self):
        frame = pd.DataFrame({
            "purchase_true": [100.0, 100.0, 100.0, 100.0],
            "purchase_pred": [100.0, 110.0, 125.0, 150.0],
            "redeem_true": [100.0, 100.0, 100.0, 100.0],
            "redeem_pred": [100.0, 100.0, 100.0, 100.0],
        })
        metrics = evaluate_prediction(frame)
        self.assertEqual(metrics["purchase_over_20_days"], 2)
        self.assertEqual(metrics["purchase_over_30_days"], 1)
        self.assertGreater(metrics["purchase_p90_ape"], metrics["purchase_median_ape"])
        self.assertGreater(metrics["risk_adjusted_loss"], metrics["flow_mape"])

    def test_missing_columns_fail_loudly(self):
        with self.assertRaises(ValueError):
            evaluate_prediction(pd.DataFrame({"purchase_true": [1]}))


if __name__ == "__main__":
    unittest.main()
