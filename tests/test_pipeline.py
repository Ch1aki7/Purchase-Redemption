"""基础与对抗性时间泄漏测试，使用标准库 unittest。"""
import unittest
import numpy as np
import pandas as pd
from src.behavior_features import aggregate_behavior
from src.preprocess import balance_consistency_check
from src.feature_engineering import make_features,future_row,validate_no_leakage
from src.models import fit_model
from src.predict import recursive_forecast
from src.metrics import evaluate,simulated_score
from src.ensemble import choose_weights,blend

def daily():
    """确定性测试数据；不会写入真实实验输出。"""
    i=pd.date_range('2014-01-01',periods=150)
    return pd.DataFrame({'total_purchase':np.arange(150)+100.,'total_redeem':np.arange(150)*2+50.,'yield_rate':np.arange(150)/100},index=i)

class FeatureTests(unittest.TestCase):
    def test_lag_and_rolling(self):
        d=daily();x=make_features(d)
        self.assertEqual(x.purchase_lag_1.iloc[40],d.total_purchase.iloc[39])
        self.assertEqual(x.purchase_rolling_mean_7.iloc[40],d.total_purchase.iloc[33:40].mean())
    def test_future_perturbation(self):
        d=daily();a=make_features(d,True); changed=d.copy();changed.iloc[80:]*=999
        pd.testing.assert_frame_equal(a.iloc[:81],make_features(changed,True).iloc[:81])
    def test_inference_matches_training(self):
        d=daily(); x=make_features(d,True)
        row=future_row(d.index[80],d.iloc[:80],list(x))
        np.testing.assert_allclose(row.to_numpy(float),x.iloc[[80]].to_numpy(float),rtol=1e-10)
    def test_overlap_is_rejected(self):
        d=daily()
        with self.assertRaisesRegex(ValueError,'重叠'):
            validate_no_leakage(d,d.index[-3:])
    def test_contaminated_feature_rejected(self):
        d=daily();x=make_features(d);x['purchase_lag_1']=d.total_purchase
        with self.assertRaisesRegex(ValueError,'特征'):
            validate_no_leakage(d,features=x)
    def test_september_rejected(self):
        d=daily();d.index=pd.date_range('2014-07-01',periods=len(d))
        with self.assertRaises(ValueError):
            validate_no_leakage(d)
    def test_missing_day_rejected(self):
        with self.assertRaises(ValueError):
            validate_no_leakage(daily().drop(daily().index[10]))

class PreprocessTests(unittest.TestCase):
    def test_aggregate_and_balance(self):
        b=pd.DataFrame({'user_id':[1,1,2],'date':pd.to_datetime(['2014-01-01','2014-01-02','2014-01-01']),'purchase':[10,20,40],'redeem':[3,5,10],'balance':[107,122,80],'previous_balance':[100,107,50]})
        p=pd.DataFrame({'user_id':[1,2],'sex':[0,1]})
        d=aggregate_behavior(b,p)
        self.assertEqual(d.total_purchase.iloc[0],50)
        self.assertEqual(d.total_redeem.iloc[0],13)
        report,samples=balance_consistency_check(b)
        self.assertEqual(report['balance_error']['anomalies'],0)
        b.loc[1,'balance']=999
        report,samples=balance_consistency_check(b)
        self.assertEqual(report['balance_error']['anomalies'],1)
        self.assertEqual(len(samples),1)

class ForecastTests(unittest.TestCase):
    def test_recursive_30_days_finite_nonnegative(self):
        d=daily();models={t:fit_model(d,t,{'kind':'Weekly'}) for t in ['purchase','redeem']}
        p=recursive_forecast(d,pd.date_range(d.index[-1]+pd.Timedelta(days=1),periods=30),models)
        self.assertEqual(len(p),30);self.assertTrue(np.isfinite(p).all().all());self.assertTrue((p>=0).all().all())
        self.assertEqual(p.purchase.iloc[7],p.purchase.iloc[0])
    def test_negative_is_counted(self):
        d=daily();d.iloc[-1,0]=-1
        m={t:fit_model(d,t,{'kind':'Yesterday'}) for t in ['purchase','redeem']}
        p=recursive_forecast(d,pd.date_range(d.index[-1]+pd.Timedelta(days=1),periods=30),m)
        self.assertEqual(p.attrs['negative_predictions']['purchase'],1)
        self.assertTrue((p>=0).all().all())
    def test_metrics_zero_and_score(self):
        a=daily().iloc[:2]*0;p=pd.DataFrame({'purchase':[0,1],'redeem':[0,1]},index=a.index)
        self.assertTrue(np.isfinite(list(evaluate(a,p).values())).all())
        np.testing.assert_equal(simulated_score([0,.3,.4]),[10,0,0])
    def test_ensemble_separate_targets(self):
        a=daily().iloc[:10]; p=a[['total_purchase','total_redeem']].rename(columns={'total_purchase':'purchase','total_redeem':'redeem'})
        q=p*2; weights=choose_weights({'perfect':p,'bad':q},a)
        np.testing.assert_allclose(blend({'perfect':p,'bad':q},weights),p)

if __name__=='__main__':
    unittest.main()
