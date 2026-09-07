"""真实多步样本边界、分项守恒、入模收据及选择期隔离测试。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.multistep_samples import build_origin,build_samples,validate_samples,COMPONENTS
from src.traced_direct import fit_traced,experiments,predict_at_origin,select_rows,row_weights
from src.sample_experiment import choose_winners

def toy_inputs():
    """只用于测试的明确合成数据，不写入真实实验结果。"""
    dates=pd.date_range('2013-07-01',periods=170);rows=[]
    for i,date in enumerate(dates):
        for user in [1,2,3]:
            direct=100+i*2+user;credit=5;consume=10+user;bal=20+i;card=30+i
            rows.append({'date':date,'user_id':user,'balance':1000+user*300+i,'purchase':direct+credit,'redeem':consume+bal+card,'direct_purchase_amt':direct,'share_amt':credit,'consume_amt':consume,'tftobal_amt':bal,'tftocard_amt':card})
    raw=pd.DataFrame(rows)
    daily=raw.groupby('date')[['purchase','redeem']].sum().rename(columns={'purchase':'total_purchase','redeem':'total_redeem'})
    for name,c in COMPONENTS.items():daily[name]=raw.groupby('date')[c].sum()
    return daily,raw

class SampleTests(unittest.TestCase):
    def test_future_perturbation_and_training_inference_match(self):
        daily,raw=toy_inputs();origin=daily.index[120];dates=daily.index[121:140]
        x,y,s,_=build_origin(daily,raw,origin,dates)
        changed=raw.copy();changed.loc[changed.date>origin,['balance','purchase','redeem']]*=1000
        # 不要求篡改后的标签守恒；只比较只读历史的推理入口。
        xx,_,ss,_=build_origin(daily.loc[:origin],changed,origin,dates,with_labels=False)
        pd.testing.assert_frame_equal(x,xx)
        pd.testing.assert_frame_equal(s,ss)
        self.assertTrue((y.purchase>0).all())
    def test_labels_and_cohorts_add_up(self):
        daily,raw=toy_inputs();origin=daily.index[120];dates=daily.index[121:130]
        _,y,s,info=build_origin(daily,raw,origin,dates)
        np.testing.assert_allclose(y.direct*s.direct+y.yield_credit*s.yield_credit,daily.loc[dates,'total_purchase'])
        np.testing.assert_allclose(sum(y[g+'_redeem']*s[g+'_redeem'] for g in ['high_balance','regular','inactive_or_unseen']),daily.loc[dates,'total_redeem'])
        self.assertLessEqual(info['threshold_source_max_date'],origin)
    def test_actual_fit_receipt_and_feature_splits(self):
        daily,raw=toy_inputs();bundle,_=build_samples(daily,raw,daily.index[-1])
        self.assertTrue(validate_samples(bundle,daily.index[-1]))
        with tempfile.TemporaryDirectory() as folder:
            spec=experiments()['C_features']
            model=fit_traced(bundle,daily.index[-1],spec,'purchase','test',Path(folder))
            receipt=model.receipts[0]
            self.assertGreater(receipt['rows_actually_passed_to_fit'],0)
            self.assertGreater(receipt['feature_groups']['component_'],0)
            self.assertEqual(receipt['model_n_features_in'],receipt['feature_count'])
            self.assertGreater(receipt['used_feature_count'],0)
            self.assertTrue(list(Path(folder).glob('*members.csv.gz')))
    def test_reject_corrupted_time_and_august_selection(self):
        daily,raw=toy_inputs();bundle,_=build_samples(daily,raw,daily.index[-1])
        bundle.meta.iloc[0,bundle.meta.columns.get_loc('feature_observed_max_date')]=pd.Timestamp('2099-01-01')
        with self.assertRaises(ValueError):validate_samples(bundle,daily.index[-1])
        with self.assertRaisesRegex(ValueError,'八月'):
            choose_winners(pd.DataFrame({'Period':['2014-08']}))

    def test_month_end_weighting_does_not_change_sample_membership(self):
        daily,raw=toy_inputs();bundle,_=build_samples(daily,raw,daily.index[-1])
        specs=experiments()
        dense=select_rows(bundle,daily.index[-1],specs['F_dense'])
        weighted=select_rows(bundle,daily.index[-1],specs['G_month_end'])
        self.assertEqual(dense.meta.sample_id.tolist(),weighted.meta.sample_id.tolist())
        mask=np.ones(len(dense.x),dtype=bool)
        a=row_weights(dense,daily.index[-1],specs['F_dense'],'purchase',mask)
        b=row_weights(weighted,daily.index[-1],specs['G_month_end'],'purchase',mask)
        self.assertTrue((a>0).all() and (b>0).all())
        self.assertFalse(np.allclose(a,b))
