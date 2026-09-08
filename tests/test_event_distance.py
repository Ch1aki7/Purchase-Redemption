import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import event_distance as event


class EventDistanceTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2013-07-01", "2014-08-31", freq="D")
        position = np.arange(len(dates), dtype=float)
        self.history = pd.DataFrame({
            "date": dates,
            "total_purchase": 180_000_000 + 15_000_000 * np.sin(position * 2 * np.pi / 7),
            "total_redeem": 160_000_000 + 12_000_000 * np.cos(position * 2 * np.pi / 7),
        })

    def test_training_writes_valid_target_isolated_submissions(self):
        root = Path(__file__).resolve().parents[1] / "output" / "test_event_distance"
        root.mkdir(parents=True, exist_ok=True)
        baseline_path = root / "baseline.csv"
        candidate_path = root / "candidate.csv"
        purchase_path = root / "purchase_only.csv"
        redeem_path = root / "redeem_only.csv"
        manifest_path = root / "manifest.json"
        baseline = pd.DataFrame({
            "report_date": range(20140901, 20140931),
            "purchase": np.arange(30) + 200_000_000,
            "redeem": np.arange(30) + 170_000_000,
        })
        baseline.to_csv(baseline_path, index=False, header=False)
        with patch.multiple(
            event,
            PREDICTION_NO_HEADER_PATH=baseline_path,
            EVENT_DISTANCE_CANDIDATE_PATH=candidate_path,
            EVENT_DISTANCE_PURCHASE_ONLY_PATH=purchase_path,
            EVENT_DISTANCE_REDEEM_ONLY_PATH=redeem_path,
            EVENT_DISTANCE_MANIFEST_PATH=manifest_path,
        ):
            result = event.train_and_export(self.history)

        self.assertEqual(result["candidate"].shape, (30, 3))
        purchase = pd.read_csv(purchase_path, header=None, names=event.SUBMISSION_COLUMNS)
        redeem = pd.read_csv(redeem_path, header=None, names=event.SUBMISSION_COLUMNS)
        self.assertTrue(purchase["redeem"].equals(baseline["redeem"]))
        self.assertTrue(redeem["purchase"].equals(baseline["purchase"]))
        self.assertTrue((purchase["purchase"] > 0).all())
        self.assertTrue((redeem["redeem"] > 0).all())
        self.assertTrue(manifest_path.exists())

    def test_rejects_invalid_target_weights(self):
        with self.assertRaisesRegex(ValueError, "权重之和"):
            event.train_and_export(
                self.history,
                {"purchase_ridge_weight": 0.8, "purchase_hw_weight": 0.3},
            )


if __name__ == "__main__":
    unittest.main()
