"""原始数据质量验证、余额核对与因果时间对齐。"""
import logging
import pandas as pd
import numpy as np
from .config import PROCESSED_DATA_DIR, OUTPUT_DIR, TRAIN_START, TRAIN_END
from .data_loader import load_tables
from .behavior_features import aggregate_behavior
from .utils import setup_logging, save_json

def balance_consistency_check(b):
    """按用户/日期验证当行昨余额与连续两日余额，分开报告非连续观测。"""
    b = b.sort_values(['user_id','date']).copy()
    g = b.groupby('user_id')
    previous = g.balance.shift(1)
    contiguous = (b.date - g.date.shift(1)).dt.days.eq(1)
    b['theoretical_from_previous_record'] = previous + b.purchase - b.redeem
    b['previous_record_error'] = (b.balance-b.theoretical_from_previous_record).where(contiguous)
    if 'previous_balance' in b:
        b['theoretical_balance'] = b.previous_balance + b.purchase-b.redeem
        b['balance_error'] = b.balance - b.theoretical_balance
        b['previous_balance_link_error'] = (b.previous_balance-previous).where(contiguous)
    else:
        b['theoretical_balance'] = b.theoretical_from_previous_record.where(contiguous)
        b['balance_error'] = b.previous_record_error
    checks = {}
    for c in ['balance_error','previous_record_error','previous_balance_link_error']:
        if c in b:
            v = b[c].dropna()
            checks[c] = {'checked':len(v),'anomalies':int(v.abs().gt(1).sum()),'ratio':float(v.abs().gt(1).mean()),'max_absolute_error':float(v.abs().max())}
    checks['nonconsecutive_or_first_records'] = int((~contiguous).sum())
    for name, cols, total in [('purchase_components',['direct_purchase_amt','share_amt'],'purchase'),('direct_components',['purchase_bal_amt','purchase_bank_amt'],'direct_purchase_amt'),('redeem_components',['consume_amt','transfer_amt'],'redeem'),('transfer_components',['tftobal_amt','tftocard_amt'],'transfer_amt')]:
        if all(c in b for c in cols+[total]):
            err=b[total]-b[cols].sum(axis=1)
            checks[name]={'anomalies':int(err.abs().gt(1).sum()),'max_absolute_error':float(err.abs().max())}
    samples=b.loc[b.balance_error.abs().gt(1) | b.previous_record_error.abs().gt(1)].head(100)
    return checks, samples

def preprocess():
    """生成每日汇总、质量报告和余额异常样本；异常不静默删除。"""
    setup_logging()
    t=load_tables()
    b,p=t['user_balance'],t['user_profile']
    issues=[]
    def record(name,count,detail):
        issues.append({'check':name,'count':int(count),'detail':detail})
        logging.log(logging.WARNING if count else logging.INFO,'%s: %s %s',name,count,detail)
    record('duplicate_user_date',b.duplicated(['user_id','date']).sum(),'主键重复时停止，避免重复聚合')
    if b.duplicated(['user_id','date']).any() or p.user_id.duplicated().any():
        raise ValueError('用户日期或画像用户主键重复，需要人工确认业务口径')
    for c in ['purchase','redeem','balance']:
        b[c]=pd.to_numeric(b[c],errors='coerce')
        if not np.isfinite(b[c]).all():
            raise ValueError(f'{c} 有缺失或非有限数值，拒绝静默填零')
        record('negative_'+c,(b[c]<0).sum(),'保留并告警；可能是冲正，应结合业务解释')
    record('unknown_profile_users',b.loc[~b.user_id.isin(p.user_id),'user_id'].nunique(),'保留交易并报告画像覆盖率')
    checks,samples=balance_consistency_check(b)
    save_json(checks,OUTPUT_DIR/'balance_consistency.json')
    samples.to_csv(OUTPUT_DIR/'balance_anomaly_samples.csv',index=False)
    logging.info('余额校验: %s',checks)
    d=aggregate_behavior(b,p)
    expected=pd.date_range(TRAIN_START,TRAIN_END)
    record('missing_transaction_dates',len(expected.difference(d.index)),'缺失平台日不可当作真实零资金流')
    if not expected.equals(d.index):
        raise ValueError('交易日期不等于配置的连续历史区间，请检查数据')
    for key in ['share_interest','bank_shibor']:
        e=t[key].set_index('date').sort_index()
        if not e.index.is_unique:
            raise ValueError(f'{key} 日期重复')
        missing=d.index.difference(e.index)
        record(key+'_missing_dates',len(missing),f'其中周末 {sum(missing.dayofweek>=5)} 天；只向前填充过去报价，开头无报价保持缺失')
        d=d.join(e.reindex(e.index.union(d.index)).sort_index().ffill().reindex(d.index))
    for c in ['total_purchase','total_redeem']:
        # 仅做描述性异常标注，绝不作为全样本清洗阈值或预测输入。
        q1,q3=d[c].quantile([.25,.75]); d[c+'_outlier']=d[c].gt(q3+3*(q3-q1))|d[c].lt(q1-3*(q3-q1))
        record(c+'_outlier_days',d[c+'_outlier'].sum(),'全历史 IQR，仅 EDA 使用，保留原值')
    d.to_csv(PROCESSED_DATA_DIR/'daily_balance.csv',index_label='date')
    pd.DataFrame(issues).to_csv(PROCESSED_DATA_DIR/'validation_report.csv',index=False)
    distributions={c:p[c].value_counts(dropna=False).to_dict() for c in p if c!='user_id'}
    save_json(distributions,OUTPUT_DIR/'user_distributions.json')
    logging.info('Phase 2/3 完成: %s 日汇总，金额逐日求和，无删除余额异常',len(d))
    return d

if __name__=='__main__':
    preprocess()
