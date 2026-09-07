"""直接多步预测与开发期选择边界的回归测试。"""
import unittest
import numpy as np
import pandas as pd
from src.peak_models import origin_features,make_direct_training,fit_calendar,forecast_new
from src.optimize import optimize,select_on_development,candidate_specs,DEVELOPMENT_END
from tests.test_pipeline import daily

class OptimizationTests(unittest.TestCase):
    def test_direct_feature_uses_origin_only(self):
        d=daily()
        origin=d.iloc[:100]
        future=d.index[100:120]
        x,_=origin_features(origin,future)
        changed=d.copy();changed.iloc[100:]*=1000
        y,_=origin_features(changed.iloc[:100],future)
        pd.testing.assert_frame_equal(x,y)
        # 对所有未来日，起点lag保持相同，未使用未来第2天的真实值。
        self.assertEqual(x.purchase_origin_lag_0.nunique(),1)
    def test_training_labels_and_origins_bounded(self):
        d=daily()
        x,y,meta=make_direct_training(d,120)
        self.assertEqual(len(x),len(y));self.assertEqual(len(x),len(meta))
        self.assertTrue((meta.origin<meta.label_date).all())
        self.assertLessEqual(meta.label_date.max(),d.index.max())
    def test_selection_rejects_august(self):
        d=daily();d.index=pd.date_range('2014-04-01',periods=len(d))
        with self.assertRaisesRegex(ValueError,'八月'):
            select_on_development(d,{})
        with self.assertRaises(ValueError):
            optimize(d)
    def test_calendar_30days_finite(self):
        d=daily();spec=candidate_specs()['Calendar_w90_a3_log0']
        models={t:fit_calendar(d,t,spec) for t in ['purchase','redeem']}
        p=forecast_new(models,d,pd.date_range(d.index[-1]+pd.Timedelta(days=1),periods=30))
        self.assertEqual(len(p),30)
        self.assertTrue(np.isfinite(p).all().all());self.assertTrue((p>=0).all().all())
