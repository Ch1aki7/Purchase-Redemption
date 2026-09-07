"""真实原始数据和交付文件集成核验。"""
import unittest
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, FORECAST_START, FORECAST_END

class ArtifactTests(unittest.TestCase):
    @unittest.skipUnless((OUTPUT_DIR/'prediction_201409.csv').exists(),'先运行完整pipeline')
    def test_saved_prediction_contract(self):
        p=pd.read_csv(OUTPUT_DIR/'prediction_201409.csv')
        self.assertEqual(list(p),['date','purchase','redeem'])
        expected=pd.date_range(FORECAST_START,FORECAST_END).strftime('%Y%m%d').astype(int).tolist()
        self.assertEqual(p.date.tolist(),expected)
        self.assertTrue(np.isfinite(p[['purchase','redeem']]).all().all())
        self.assertTrue((p[['purchase','redeem']]>=0).all().all())
        self.assertTrue(all(pd.api.types.is_integer_dtype(p[c]) for c in ['purchase','redeem']))
    @unittest.skipUnless((RAW_DATA_DIR/'user_balance_table.csv').exists(),'需要原始CSV')
    def test_actual_aggregation_matches_raw(self):
        raw=pd.read_csv(RAW_DATA_DIR/'user_balance_table.csv',usecols=['report_date','total_purchase_amt','total_redeem_amt'])
        expected=raw.groupby('report_date')[['total_purchase_amt','total_redeem_amt']].sum()
        actual=pd.read_csv(PROCESSED_DATA_DIR/'daily_balance.csv')
        np.testing.assert_array_equal(expected.to_numpy(),actual[['total_purchase','total_redeem']].to_numpy())
    @unittest.skipUnless((OUTPUT_DIR/'validation_predictions.csv').exists(),'先运行回测')
    def test_actual_holdout_months(self):
        p=pd.read_csv(OUTPUT_DIR/'validation_predictions.csv')
        self.assertEqual(set(pd.to_datetime(p.date).dt.month),{6,7,8})
        self.assertFalse(p.filter(regex='actual_|pred_').isna().any().any())
        for (_,fold),part in p.groupby(['Model','Fold']):
            self.assertEqual(len(part),30 if fold==1 else 31)
