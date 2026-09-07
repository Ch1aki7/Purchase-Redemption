"""所有目标派生特征严格先移位，训练和递归推理共享定义。"""
import logging
import numpy as np
import pandas as pd
from .calendar_features import calendar_features
from .config import LAG_FEATURES, ROLLING_WINDOWS, TARGETS, TRAIN_END, PROCESSED_DATA_DIR
EXTRA_COLUMNS=['daily_active_users','purchase_user_ratio','redeem_user_ratio','avg_balance','active_sex_1_ratio','new_observed_user_ratio','yield_rate','yield_7d','shibor_overnight','shibor_1w','shibor_1m']

def make_features(d, enriched=False):
    """向量化历史特征，当前日金额/行为/报价均不作为当前输入。"""
    x=calendar_features(d.index)
    for target in TARGETS:
        s=d['total_'+target]
        for lag in LAG_FEATURES:
            x[f'{target}_lag_{lag}']=s.shift(lag)
        past=s.shift(1)
        for w in ROLLING_WINDOWS:
            for stat in ['mean','min','max','median']:
                x[f'{target}_rolling_{stat}_{w}']=getattr(past.rolling(w),stat)()
            if w>=7:
                x[f'{target}_rolling_std_{w}']=past.rolling(w).std()
    if enriched:
        for c in EXTRA_COLUMNS:
            if c in d:
                x[c+'_lag_1']=pd.to_numeric(d[c],errors='coerce').shift(1)
    return x

def future_row(date, history, columns):
    """仅传入预测日之前的历史；多步行为/利率保持起点最后已知值。"""
    if history.index.max()>=date:
        raise ValueError('构造未来特征时包含当日或未来观测')
    row=calendar_features([date]).iloc[0].to_dict()
    for t in TARGETS:
        a=history['total_'+t].to_numpy(dtype=float)
        for lag in LAG_FEATURES:
            row[f'{t}_lag_{lag}']=a[-lag] if len(a)>=lag else np.nan
        for w in ROLLING_WINDOWS:
            v=a[-w:]
            for name,func in [('mean',np.mean),('min',np.min),('max',np.max),('median',np.median)]:
                row[f'{t}_rolling_{name}_{w}']=float(func(v)) if len(a)>=w else np.nan
            if w>=7:
                row[f'{t}_rolling_std_{w}']=float(np.std(v,ddof=1)) if len(a)>=w else np.nan
    for c in EXTRA_COLUMNS:
        if c in history:
            row[c+'_lag_1']=history[c].iloc[-1]
    return pd.DataFrame([row],index=[date]).reindex(columns=columns)

def validate_no_leakage(history, validation_index=None, features=None, enriched=False):
    """验证边界、连续日期、lag/rolling 和事后行为/外生变量的移位。"""
    errors=[]
    if not history.index.is_unique or not history.index.is_monotonic_increasing:
        errors.append('历史索引未严格递增且唯一')
    if not history.index.equals(pd.date_range(history.index.min(),history.index.max())):
        errors.append('历史日期不连续，lag 不能解释为天')
    if history.index.max()>pd.Timestamp(TRAIN_END):
        errors.append('2014-09 或更晚数据进入历史')
    if validation_index is not None and history.index.max()>=pd.DatetimeIndex(validation_index).min():
        errors.append('train/validation 时间重叠')
    if features is not None:
        expected=make_features(history,enriched)
        if list(features)!=list(expected) or not np.allclose(features.to_numpy(float),expected.to_numpy(float),equal_nan=True):
            errors.append('特征不符合严格滞后定义，可能含当日或未来信息')
    if errors:
        logging.error('数据泄漏警告: %s',errors)
        raise ValueError('; '.join(errors))
    return {'status':'passed','train_end':str(history.index.max()),'checks':['时间递增/连续','严格时间边界','九月隔离','特征移位（传入特征时逐值检查）'],'limitations':'程序无法证明原始报价修订时点；保守滞后一天并固定预测起点报价。静态画像假定在历史可用。'}

if __name__=='__main__':
    d=pd.read_csv(PROCESSED_DATA_DIR/'daily_balance.csv',parse_dates=['date'],index_col='date')
    x=make_features(d,True)
    validate_no_leakage(d,features=x,enriched=True)
    x.to_csv(PROCESSED_DATA_DIR/'features.csv',index_label='date')
