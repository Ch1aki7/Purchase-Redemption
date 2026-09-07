"""事前日历与起点条件直接多步预测，不使用未来目标构造特征。"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from .calendar_features import calendar_features
from .config import TARGETS,RANDOM_STATE
from .feature_engineering import validate_no_leakage

def calendar_design(dates,reference,style='day'):
    """公历、月内工作日位置和交互；工作日仅指周一至周五，非节假日历。"""
    dates=pd.DatetimeIndex(dates)
    x=calendar_features(dates).astype(float)
    x=x.drop(columns=['year','week_of_year'])
    for w in range(7):
        x[f'weekday_{w}']=(dates.dayofweek==w).astype(float)
    if style=='day':
        for day in range(1,32):
            x[f'dom_{day}']=(dates.day==day).astype(float)
    else:
        for k in range(1,5):
            x[f'dom_sin_{k}']=np.sin(2*np.pi*k*(dates.day-1)/dates.days_in_month)
            x[f'dom_cos_{k}']=np.cos(2*np.pi*k*(dates.day-1)/dates.days_in_month)
    for week in range(5):
        for wd in range(7):
            x[f'week_{week}_wd_{wd}']=(((dates.day-1)//7==week)&(dates.dayofweek==wd)).astype(float)
    starts=dates.to_period('M').to_timestamp()
    x['weekday_index_in_month']=np.busday_count(starts.values.astype('datetime64[D]'),dates.values.astype('datetime64[D]'))
    x['elapsed_months']=(dates-pd.Timestamp(reference)).days/30
    # 连续整数缩放，避免它们支配独热日期特征的正则化。
    for c in ['day','day_of_month','days_to_month_end','days_from_month_start','weekday_index_in_month']:
        x[c]/=31
    x['month']/=12
    return x

@dataclass
class CalendarModel:
    """近期加权的月内/星期回归，线性或对数目标可选。"""
    spec: dict
    target: str
    estimator: object
    reference: object
    scale: float
    columns: list

    def forecast(self,history,dates):
        """只用已拟合参数与未来公历，不递归回填平滑值。"""
        x=calendar_design(dates,self.reference,self.spec['style'])
        x=x[self.columns]
        y=self.estimator.predict(x)
        return np.exp(y)*self.scale if self.spec['log'] else y*self.scale

def fit_calendar(history,target,spec):
    """窗口、半衰期和正则化均在开发折选择。"""
    h=history.iloc[-spec['window']:]
    y=h['total_'+target].to_numpy(float)
    scale=max(float(y[-56:].mean()),1)
    x=calendar_design(h.index,h.index[-1],spec['style'])
    if not spec.get('trend',False):
        x=x.drop(columns='elapsed_months')
    age=(h.index[-1]-h.index).days.to_numpy()
    weight=np.power(.5,age/spec['half_life'])
    est=Ridge(alpha=spec['alpha'])
    est.fit(x,np.log(np.maximum(y,1)/scale) if spec['log'] else y/scale,sample_weight=weight)
    return CalendarModel(spec,target,est,h.index[-1],scale,list(x))

def origin_features(history,dates):
    """起点状态 + 各预测日的历史同星期/月日水平，所有观测严格早于dates。"""
    dates=pd.DatetimeIndex(dates)
    origin=history.index[-1]
    if dates.min()<=origin:
        raise ValueError('Direct 特征含起点之前的目标日')
    x=calendar_design(dates,origin,'fourier').drop(columns='elapsed_months').copy()
    x['horizon']=(dates-origin).days/31
    scales={}
    for t in TARGETS:
        s=history['total_'+t]
        scale=max(float(s.iloc[-28:].mean()),1)
        scales[t]=scale
        for lag in [0,1,2,6,13,20,27]:
            x[f'{t}_origin_lag_{lag}']=s.iloc[-lag-1]/scale
        for w in [7,14,28,56]:
            v=s.iloc[-w:]
            x[f'{t}_mean_{w}']=v.mean()/scale
            x[f'{t}_std_{w}']=v.std()/scale
            x[f'{t}_max_{w}']=v.max()/scale
        for w in [28,56,84]:
            v=s.iloc[-w:]
            by_week=v.groupby(v.index.dayofweek).mean()/scale
            x[f'{t}_target_weekday_{w}']=[by_week.get(i,1) for i in dates.dayofweek]
        # 月内效应先消除各历史时点的局部水平；不使用预测日之后的数值。
        normalized=(s/s.shift(1).rolling(28).mean()).dropna().iloc[-180:]
        by_dom=normalized.groupby(normalized.index.day).mean()
        count=normalized.groupby(normalized.index.day).count()
        x[f'{t}_target_dom_shrunk']=[(by_dom.get(day,1)*count.get(day,0)+3)/(count.get(day,0)+3) for day in dates.day]
    return x,scales

def make_direct_training(history,window=180,stride=7):
    """历史起点与标签均在训练边界内；起点之后只允许事前日历。"""
    xs=[];ys=[];metadata=[]
    first=max(84,len(history)-window-31)
    for pos in range(first,len(history)-7,stride):
        dates=history.index[pos+1:min(pos+32,len(history))]
        x,scales=origin_features(history.iloc[:pos+1],dates)
        y=pd.DataFrame({t:history.loc[dates,'total_'+t]/scales[t] for t in TARGETS},index=dates)
        xs.append(x);ys.append(y)
        metadata.append(pd.DataFrame({'origin':history.index[pos],'label_date':dates}))
    if not xs:
        raise ValueError('直接多步训练至少需要100天历史')
    return pd.concat(xs),pd.concat(ys),pd.concat(metadata,ignore_index=True)

@dataclass
class DirectModel:
    """共享预测期条件的单目标直接多步模型。"""
    spec: dict
    target: str
    estimator: object
    columns: list

    def forecast(self,history,dates):
        """未来各日单独由同一已知起点生成，不依赖未来真值或预测回填。"""
        x,scales=origin_features(history,dates)
        return self.estimator.predict(x[self.columns])*scales[self.target]

def fit_direct(history,target,spec,prepared=None):
    """直接多步学习以相对局部水平为目标，避免平台增长主导尺度。"""
    x,y,meta=make_direct_training(history,spec['window']) if prepared is None else prepared
    if (meta.label_date>history.index.max()).any() or (meta.origin>=meta.label_date).any():
        raise ValueError('直接多步训练标签越过边界')
    age=(history.index.max()-pd.DatetimeIndex(meta.label_date)).days.to_numpy()
    weights=np.power(.5,age/spec['half_life'])
    if spec['estimator']=='ridge':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        est=make_pipeline(StandardScaler(),Ridge(alpha=spec['alpha']))
        est.fit(x,y[target],ridge__sample_weight=weights)
    else:
        est=HistGradientBoostingRegressor(max_iter=spec['iterations'],learning_rate=.04,max_leaf_nodes=spec['leaves'],min_samples_leaf=spec['leaf'],l2_regularization=spec['l2'],early_stopping=False,random_state=RANDOM_STATE)
        est.fit(x,y[target],sample_weight=weights)
    return DirectModel(spec,target,est,list(x))

def forecast_new(models,history,dates):
    """统一直接预测数值校验与负数截断计数。"""
    dates=pd.DatetimeIndex(dates)
    validate_no_leakage(history,dates)
    if not dates.equals(pd.date_range(history.index.max()+pd.Timedelta(days=1),periods=len(dates))):
        raise ValueError('未来日期必须从历史下一天连续开始')
    pred=pd.DataFrame(index=dates);counts={}
    for t,m in models.items():
        y=np.asarray(m.forecast(history,dates),float)
        if not np.isfinite(y).all():
            raise ValueError('模型产生非有限预测')
        counts[t]=int((y<0).sum());pred[t]=np.maximum(0,y)
    pred.attrs['negative_predictions']=counts
    return pred
