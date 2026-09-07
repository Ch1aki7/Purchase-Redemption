"""实际样本入模、特征与采样方式消融、开发期模型选择。"""
from dataclasses import dataclass
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from .config import TARGETS,RANDOM_STATE,MODEL_DIR,OUTPUT_DIR
from .multistep_samples import SampleBundle,COMPONENT_TARGET,GROUPS,build_origin,validate_samples,frame_hash

def experiments():
    """事先固定逐项对照关系，避免把多个改动的收益混为一谈。"""
    common={'mode':'total','features':'core','origins':'weekly','loss':'squared_error','relative':False,'horizon_split':False,'window':210}
    specs={'A_core_weekly':dict(common)}
    specs['B_absolute']={**common,'loss':'absolute_error'}
    specs['C_features']={**specs['B_absolute'],'features':'all'}
    specs['D_components']={**specs['C_features'],'mode':'components'}
    specs['E_cohorts']={**specs['C_features'],'mode':'cohorts'}
    specs['F_dense']={**specs['C_features'],'origins':'dense'}
    specs['G_month_end']={**specs['F_dense'],'origins':'month_end_weighted'}
    specs['H_relative']={**specs['F_dense'],'relative':True}
    specs['I_component_relative']={**specs['H_relative'],'mode':'components'}
    specs['J_cohort_relative']={**specs['H_relative'],'mode':'cohorts'}
    specs['K_horizon_groups']={**specs['H_relative'],'horizon_split':True}
    specs['L_recent120']={**specs['H_relative'],'window':120}
    specs['M_recent60']={**specs['H_relative'],'window':60}
    return specs

def feature_columns(x,spec):
    """core与扩展列显式分开并记录真正传给fit的列。"""
    prefixes=('component_','behavior_','cohort_')
    return [c for c in x if spec['features']=='all' or not c.startswith(prefixes)]

def select_rows(bundle,cutoff,spec,drop_group=None):
    """从公共样本池选取当折合法样本；样本规则只依赖起点和日期。"""
    m=bundle.meta;cutoff=pd.Timestamp(cutoff)
    mask=(m.label_date<=cutoff)&(m.origin>=cutoff-pd.Timedelta(days=spec['window']))
    if spec['origins']=='weekly':
        mask &= m.weekly_origin
    elif spec['origins'] in ['dense','month_end_weighted']:
        mask &= m.dense_origin | m.origin_month_end
    if drop_group:
        age=(cutoff-m.origin).dt.days
        groups={'recent_0_30':age<=30,'middle_31_90':(age>30)&(age<=90),'older_91_plus':age>90,'month_end_origins':m.origin_month_end}
        mask &= ~groups[drop_group]
    out=bundle.subset(mask)
    validate_samples(out,cutoff)
    return out

def heads_for(target,mode):
    """分项和群体分别预测后严格相加，不重复计入资金。"""
    if mode=='total':
        return [target]
    if mode=='components':
        return [c for c,t in COMPONENT_TARGET.items() if t==target]
    return [g+'_'+target for g in GROUPS]

def row_weights(bundle,cutoff,spec,head,mask):
    """纠正同一真实日期被多个起点重复采样；相对损失权重仅由训练标签计算。"""
    m=bundle.meta.loc[mask]
    duplicate=m.groupby('label_date').sample_id.transform('count').to_numpy(float)
    age=(pd.Timestamp(cutoff)-m.label_date).dt.days.to_numpy()
    w=np.power(.5,age/90)/duplicate
    if spec['origins']=='month_end_weighted':
        w*=np.where(m.origin_month_end,3.,1.)
    if spec['relative']:
        y=bundle.y.loc[mask,head].to_numpy(float)
        # 对极小分项限制权重，以免接近0的记录主导整个模型。
        floor=max(float(np.quantile(y,.1)),.05)
        w/=np.maximum(np.abs(y),floor)
    return w/w.mean()

@dataclass
class TracedDirectModel:
    """保留每个真实拟合的特征、imputer、模型与样本摘要。"""
    target: str
    spec: dict
    columns: list
    models: dict
    receipts: list

    def predict(self,x,scales,horizons):
        """从预测起点特征一次生成所有horizon，不使用未来真值。"""
        p=np.zeros(len(x));negative_counts={}
        for (head,lo,hi),(imputer,est) in self.models.items():
            mask=(horizons>=lo)&(horizons<=hi)
            if not mask.any():
                continue
            values=est.predict(imputer.transform(x.loc[mask,self.columns]))*scales.loc[mask,head].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError('分项原始预测非有限，拒绝静默截断')
            negative_counts[f'{head}_{lo}_{hi}']=int((values<0).sum())
            p[mask]+=np.maximum(0,values)
        if not np.isfinite(p).all():
            raise ValueError('多步模型输出非有限')
        self.last_negative_counts=negative_counts
        return p

def fit_traced(bundle,cutoff,spec,target,name,receipt_dir,drop_group=None):
    """每次fit都记录实际矩阵、样本ID、目标、权重和模型参数。"""
    b=select_rows(bundle,cutoff,spec,drop_group)
    if len(b.x)<40:
        raise ValueError('消融后训练样本不足')
    columns=feature_columns(b.x,spec)
    groups=[(1,7),(8,14),(15,31)] if spec['horizon_split'] else [(1,31)]
    fitted={};receipts=[];members=[]
    for head in heads_for(target,spec['mode']):
        for lo,hi in groups:
            mask=(b.meta.horizon>=lo)&(b.meta.horizon<=hi)
            x=b.x.loc[mask,columns];y=b.y.loc[mask,head]
            imputer=SimpleImputer(strategy='median',keep_empty_features=True)
            xx=imputer.fit_transform(x)
            weights=row_weights(b,cutoff,spec,head,mask)
            est=HistGradientBoostingRegressor(loss=spec['loss'],max_iter=120,learning_rate=.04,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=3,early_stopping=False,random_state=RANDOM_STATE)
            est.fit(xx,y,sample_weight=weights)
            split_counts=np.zeros(len(columns),dtype=int)
            for stage in est._predictors:
                for tree in stage:
                    nodes=tree.nodes
                    ids=nodes['feature_idx'][nodes['is_leaf']==0]
                    split_counts+=np.bincount(ids.astype(int),minlength=len(columns))
            used_features={c:int(n) for c,n in zip(columns,split_counts) if n>0}
            key=(head,lo,hi);fitted[key]=(imputer,est)
            fit_id=f'{name}_{target}_{head}_{lo}_{hi}'
            receipt={'fit_id':fit_id,'experiment':name,'target':target,'head':head,'horizon_min':lo,'horizon_max':hi,'train_cutoff':str(pd.Timestamp(cutoff).date()),'rows_actually_passed_to_fit':len(x),'unique_label_dates':b.meta.loc[mask,'label_date'].nunique(),'unique_origins':b.meta.loc[mask,'origin'].nunique(),'feature_count':len(columns),'feature_columns':columns,'x_hash':frame_hash(x),'y_hash':frame_hash(y.to_frame()),'model_n_features_in':int(est.n_features_in_),'model_iterations':int(est.n_iter_),'feature_groups':{p:sum(c.startswith(p) for c in columns) for p in ['component_','behavior_','cohort_']},'used_split_features':used_features,'used_feature_count':len(used_features),'drop_group':drop_group,'spec':spec}
            receipts.append(receipt)
            member=b.meta.loc[mask,['sample_id','origin','horizon','label_date','feature_observed_max_date']].copy()
            member['fit_id']=fit_id;member['sample_weight']=weights
            members.append(member)
    receipt_dir.mkdir(parents=True,exist_ok=True)
    stem=f'{name}_{target}'+('_drop_'+drop_group if drop_group else '')
    (receipt_dir/(stem+'.json')).write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf8')
    pd.concat(members).to_csv(receipt_dir/(stem+'_members.csv.gz'),index=False,compression='gzip')
    return TracedDirectModel(target,spec,columns,fitted,receipts)

def predict_at_origin(models,daily,raw,origin,dates):
    """显式传历史到特征构造，未来明细不得进入预测接口。"""
    dates=pd.DatetimeIndex(dates);origin=pd.Timestamp(origin)
    if not 1<=len(dates)<=31 or not dates.equals(pd.date_range(origin+pd.Timedelta(days=1),periods=len(dates))):
        raise ValueError('直接多步预测必须从起点下一天开始，连续1至31天')
    history=daily.loc[:origin]
    observed_raw=raw.loc[raw.date<=origin]
    x,_,scales,_=build_origin(history,observed_raw,pd.Timestamp(origin),pd.DatetimeIndex(dates),with_labels=False)
    horizons=(pd.DatetimeIndex(dates)-pd.Timestamp(origin)).days.to_numpy()
    p=pd.DataFrame(index=pd.DatetimeIndex(dates))
    for t,model in models.items():
        p[t]=model.predict(x,scales,horizons)
    p.attrs['negative_raw_predictions']={t:m.last_negative_counts for t,m in models.items()}
    return p
