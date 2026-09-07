"""可追溯的多步训练样本：总额、交易分项与起点确定的用户群。"""
from dataclasses import dataclass
import hashlib
import logging
import numpy as np
import pandas as pd
from .config import OUTPUT_DIR,PROCESSED_DATA_DIR,TARGETS
from .data_loader import discover,map_fields
from .peak_models import origin_features

SAMPLE_DIR=OUTPUT_DIR/'multistep'
COMPONENTS={'direct':'direct_purchase_amt','yield_credit':'share_amt','consume':'consume_amt','to_balance':'tftobal_amt','to_card':'tftocard_amt'}
COMPONENT_TARGET={'direct':'purchase','yield_credit':'purchase','consume':'redeem','to_balance':'redeem','to_card':'redeem'}
GROUPS=['high_balance','regular','inactive_or_unseen']
ANCHOR=pd.Timestamp('2013-07-01')

@dataclass
class SampleBundle:
    """模型真正使用的矩阵与对应行账本，sample_id为共同索引。"""
    x: pd.DataFrame
    y: pd.DataFrame
    scales: pd.DataFrame
    meta: pd.DataFrame

    def subset(self,mask):
        """保持输入、标签、缩放和账本严格同行。"""
        return SampleBundle(*(v.loc[mask].copy() for v in [self.x,self.y,self.scales,self.meta]))

def frame_hash(frame):
    """矩阵、列名及行顺序的摘要，用于训练追踪。"""
    h=hashlib.sha256()
    h.update('|'.join(map(str,frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
    return h.hexdigest()

def load_inputs():
    """保留原始明细字段；扩展聚合不改变现有每日总额文件。"""
    raw=pd.read_csv(discover()['user_balance'])
    raw,_=map_fields(raw,'user_balance')
    raw['date']=pd.to_datetime(raw.date.astype(str),format='%Y%m%d')
    raw=raw.sort_values(['date','user_id'])
    daily=pd.read_csv(PROCESSED_DATA_DIR/'daily_balance.csv',parse_dates=['date'],index_col='date')
    for name,col in COMPONENTS.items():
        if col not in raw:
            raise ValueError(f'交易分项需要实际字段 {col}')
        daily[name]=raw.groupby('date')[col].sum()
    np.testing.assert_allclose(daily.direct+daily.yield_credit,daily.total_purchase,rtol=0,atol=1)
    np.testing.assert_allclose(daily.consume+daily.to_balance+daily.to_card,daily.total_redeem,rtol=0,atol=1)
    return daily,raw

def cohort_snapshot(raw,origin):
    """只用起点前56天决定用户组；余额为最后观测，不冒充当天实际余额。"""
    history=raw[(raw.date<=origin)&(raw.date>origin-pd.Timedelta(days=56))]
    latest=history.sort_values('date').drop_duplicates('user_id',keep='last').set_index('user_id')
    threshold=float(latest.balance.quantile(.8))
    labels=pd.Series('regular',index=latest.index)
    labels.loc[latest.balance>=threshold]='high_balance'
    info={'origin':origin,'threshold_last_observed_balance':threshold,'threshold_source_max_date':history.date.max(),'observed_users':len(latest),'stale_over_7days_users':int(((origin-latest.date).dt.days>7).sum())}
    return labels,info

def cohort_series(raw,origin,dates):
    """训练标签可在起点之后；用户归属完全由起点已知历史决定。"""
    labels,info=cohort_snapshot(raw,origin)
    start=origin-pd.Timedelta(days=55)
    relevant=raw[(raw.date>=start)&(raw.date<=dates.max())].copy()
    relevant['cohort']=relevant.user_id.map(labels).fillna('inactive_or_unseen')
    grouped=relevant.groupby(['date','cohort'])[['purchase','redeem']].sum()
    idx=pd.date_range(start,dates.max())
    series={}
    for group in GROUPS:
        for t in TARGETS:
            name=f'{group}_{t}'
            series[name]=grouped[t].unstack('cohort').reindex(index=idx,columns=GROUPS).fillna(0)[group]
    return pd.DataFrame(series,index=idx),info

def build_origin(daily,raw,origin,dates,with_labels=True):
    """未来日历可用，所有观测型特征截止origin；预测模式不读取未来明细。"""
    history=daily.loc[:origin]
    if history.index.max()!=origin or pd.DatetimeIndex(dates).min()<=origin:
        raise ValueError('起点与预测日期无效')
    x,base_scales=origin_features(history,dates)
    extra={};scales={}
    for t in TARGETS:
        scales[t]=base_scales[t]
    for name in COMPONENTS:
        series=history[name]
        scale=max(float(series.tail(28).mean()),1)
        scales[name]=scale
        for w in [7,28,56]:
            extra[f'component_{name}_mean_{w}']=float(series.tail(w).mean()/scale)
        extra[f'component_{name}_level']=np.log1p(scale)
        extra[f'component_{name}_share']=scale/base_scales[COMPONENT_TARGET[name]]
        extra[f'component_{name}_cv28']=float(series.tail(28).std()/scale)
    for name in ['daily_active_users','purchase_user_ratio','redeem_user_ratio','avg_balance','new_observed_user_ratio','active_sex_1_ratio','yield_rate','shibor_overnight']:
        if name in history:
            series=history[name].astype(float)
            denom=max(abs(float(series.tail(28).mean())),1e-6)
            # 外部报价只使用前日及更早信息；其他日结数据可使用起点当天。
            if name in ['yield_rate','shibor_overnight']:
                series=series.shift(1)
                denom=max(abs(float(series.tail(28).mean())),1e-6)
            extra[f'behavior_{name}_level']=float(series.tail(28).mean())
            extra[f'behavior_{name}_change7_28']=float(series.tail(7).mean()/denom)
    # 推理时构造群体特征只传入历史明细，避免未来用户身份影响阈值。
    cohort_raw=raw if with_labels else raw.loc[raw.date<=origin]
    cohorts,info=cohort_series(cohort_raw,origin,pd.DatetimeIndex(dates) if with_labels else pd.DatetimeIndex([origin]))
    for name in cohorts:
        past=cohorts.loc[:origin,name]
        scale=max(float(past.tail(28).mean()),1)
        scales[name]=scale
        extra[f'cohort_{name}_level']=np.log1p(scale)
        extra[f'cohort_{name}_trend']=float(past.tail(7).mean()/scale)
        extra[f'cohort_{name}_cv28']=float(past.tail(28).std()/scale)
        target='purchase' if name.endswith('_purchase') else 'redeem'
        extra[f'cohort_{name}_share']=scale/base_scales[target]
    x=pd.concat([x,pd.DataFrame(extra,index=x.index)],axis=1)
    # 唯一安全的NaN处理是训练期拟合的imputer；这里不做未来填充。
    x=x.replace([np.inf,-np.inf],np.nan)
    ys=pd.DataFrame(index=pd.DatetimeIndex(dates))
    if with_labels:
        for t in TARGETS:
            ys[t]=daily.loc[dates,'total_'+t]/scales[t]
        for name in COMPONENTS:
            ys[name]=daily.loc[dates,name]/scales[name]
        for name in cohorts:
            ys[name]=cohorts.loc[dates,name]/scales[name]
        for t in TARGETS:
            rebuilt=sum(ys[g+'_'+t]*scales[g+'_'+t] for g in GROUPS)
            np.testing.assert_allclose(rebuilt,daily.loc[dates,'total_'+t],rtol=0,atol=1)
    return x,ys,pd.DataFrame(scales,index=x.index),info

def build_samples(daily,raw,cutoff):
    """生成固定anchor的密集/每周/月末起点合集，每条样本可重建。"""
    cutoff=pd.Timestamp(cutoff)
    daily=daily.loc[:cutoff].copy();raw=raw.loc[raw.date<=cutoff].copy()
    if daily.index.max()!=cutoff:
        raise ValueError('训练截止日缺少数据')
    all_dates=daily.index[90:-1]
    origins=[d for d in all_dates if (d-ANCHOR).days%3==0 or (d-ANCHOR).days%7==0 or d.is_month_end]
    chunks=[[],[],[],[]];snapshots=[]
    for number,origin in enumerate(origins,1):
        dates=pd.date_range(origin+pd.Timedelta(days=1),min(origin+pd.Timedelta(days=31),cutoff))
        x,y,scales,info=build_origin(daily,raw,origin,dates)
        ids=[f'{origin:%Y%m%d}_h{h:02d}' for h in range(1,len(dates)+1)]
        meta=pd.DataFrame({'sample_id':ids,'origin':origin,'horizon':range(1,len(dates)+1),'label_date':dates,'feature_observed_max_date':origin,'calendar_known_in_advance':True,'origin_month_end':origin.is_month_end,'weekly_origin':(origin-ANCHOR).days%7==0,'dense_origin':(origin-ANCHOR).days%3==0},index=ids)
        for frame in [x,y,scales]:
            frame.index=ids;frame.index.name='sample_id'
        chunks[0].append(x);chunks[1].append(y);chunks[2].append(scales);chunks[3].append(meta)
        snapshots.append(info)
        if number%20==0:
            logging.info('样本构造 %s/%s origin=%s',number,len(origins),origin.date())
    bundle=SampleBundle(*(pd.concat(parts) for parts in chunks))
    validate_samples(bundle,cutoff)
    return bundle,pd.DataFrame(snapshots)

def validate_samples(bundle,cutoff):
    """特征来源<=起点<标签<=训练截止；矩阵与账本必须一致。"""
    m=bundle.meta
    if not all(bundle.x.index.equals(v.index) for v in [bundle.y,bundle.scales,m]):
        raise ValueError('输入标签或账本行错位')
    if not m.index.is_unique:
        raise ValueError('sample_id重复')
    if (m.feature_observed_max_date>m.origin).any() or (m.origin>=m.label_date).any() or (m.label_date>pd.Timestamp(cutoff)).any():
        raise ValueError('多步训练样本时间边界泄漏')
    if not ((m.label_date-m.origin).dt.days.to_numpy()==m.horizon.to_numpy()).all():
        raise ValueError('horizon与日期差不一致')
    if not np.isfinite(bundle.y.to_numpy()).all() or not np.isfinite(bundle.scales.to_numpy()).all():
        raise ValueError('标签或缩放非有限')
    return True

def save_bundle(bundle,snapshots,folder):
    """保存实际输入、标签、缩放和每行来源，可独立检查与复现。"""
    folder.mkdir(parents=True,exist_ok=True)
    bundle.x.to_csv(folder/'features.csv.gz',compression='gzip',index_label='sample_id')
    bundle.y.to_csv(folder/'normalized_labels.csv.gz',compression='gzip',index_label='sample_id')
    bundle.scales.to_csv(folder/'origin_scales.csv.gz',compression='gzip',index_label='sample_id')
    bundle.meta.to_csv(folder/'sample_ledger.csv',index=False)
    snapshots.to_csv(folder/'cohort_thresholds.csv',index=False)
