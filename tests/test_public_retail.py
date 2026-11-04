"""時点混入・欠品フラグの向き・未到着時の停止を検査する。"""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

path = Path(__file__).resolve().parents[1] / "16_cases/06_public_retail.py"
spec = importlib.util.spec_from_file_location("public_retail", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublicRetailTest(unittest.TestCase):
    def setUp(self):
        self.date = pd.Timestamp("2024-06-01")
        self.frame = pd.DataFrame({
            "product_id": 1,
            "dt": pd.date_range(self.date - pd.Timedelta(days=28), periods=30),
            "hours_sale": [np.ones(24) for _ in range(30)],
            "hours_stock_status": [np.zeros(24, dtype=int) for _ in range(30)]})

    def test_today_and_future_do_not_change_prediction(self):
        a = module.predict_day(self.frame, self.date, 28, "mean")
        for i in self.frame.index[self.frame.dt >= self.date]:
            self.frame.at[i, "hours_sale"] = np.full(24, 1e9)
            self.frame.at[i, "hours_stock_status"] = np.ones(24, dtype=int)
        self.assertEqual(a, module.predict_day(self.frame, self.date, 28, "mean"))

    def test_one_means_stockout_and_is_excluded(self):
        i = self.frame.index[self.frame.dt == self.date - pd.Timedelta(days=7)][0]
        self.frame.at[i, "hours_sale"] = np.zeros(24)
        self.frame.at[i, "hours_stock_status"] = np.ones(24, dtype=int)
        self.assertEqual(module.predict_day(self.frame, self.date, 28, "mean"), 16)

    def test_missing_yesterday_stops_automatic_run(self):
        history = self.frame[self.frame.dt < self.date - pd.Timedelta(days=1)]
        targets = self.frame[self.frame.dt == self.date]
        with self.assertRaisesRegex(ValueError, "未到着"):
            module.forecasts(history, targets, 28, "mean")


if __name__ == "__main__":
    unittest.main()
