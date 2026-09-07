"""统一模型工厂：基线、独立目标回归及 Holt-Winters。"""
import logging
from dataclasses import dataclass
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from .config import RANDOM_STATE
from .feature_engineering import make_features
SCALE=1e8
BASELINES=['Yesterday','Weekly','MA7','MA14','MA30','SameWeekday']

@dataclass
class FittedModel:
    """可序列化的单目标模型及其输入特征定义。"""
    name: str
    target: str
    estimator: object=None
    columns: object=None
    resolved_name: str=''
    enriched: bool=False

def model_specs():
    """预先声明候选集合；赎回额外比较更平滑的树参数。"""
    specs={name:{'kind':name} for name in BASELINES}
    specs.update({'Ridge':{'kind':'Ridge','alpha':30},'RandomForest':{'kind':'RandomForest','n_estimators':180,'max_depth':5,'min_samples_leaf':8},'GradientBoosting':{'kind':'GradientBoosting','n_estimators':120,'max_depth':2,'learning_rate':.03,'min_samples_leaf':10},'HistGradientBoosting':{'kind':'HistGradientBoosting','n_estimators':120,'max_depth':3,'learning_rate':.04,'min_samples_leaf':15},'RandomForest_enriched':{'kind':'RandomForest','enriched':True,'n_estimators':180,'max_depth':5,'min_samples_leaf':10},'Redeem_smooth':{'kind':'GradientBoosting','n_estimators':180,'max_depth':2,'learning_rate':.025,'min_samples_leaf':20},'TimeSeries':{'kind':'TimeSeries'}})
    return specs

def fit_model(history,target,spec):
    """所有拟合（包括 imputer/scaler）仅接收训练期数据。"""
    kind=spec['kind']; enriched=spec.get('enriched',False)
    m=FittedModel(kind,target,resolved_name=kind,enriched=enriched)
    if kind in BASELINES:
        return m
    if kind=='TimeSeries':
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            logging.warning('statsmodels 不可用，TimeSeries fallback 为 Weekly')
            m.name=m.resolved_name='Weekly (fallback)'
            return m
        m.estimator=ExponentialSmoothing(history['total_'+target]/SCALE,trend='add',damped_trend=True,seasonal='add',seasonal_periods=7,initialization_method='estimated').fit(optimized=True)
        return m
    x=make_features(history,enriched).iloc[30:]
    y=history['total_'+target].iloc[30:]/SCALE
    if len(x)<60:
        raise ValueError('至少需要90天训练数据')
    n=int(spec.get('n_estimators',120)); depth=int(spec.get('max_depth',3)); leaf=int(spec.get('min_samples_leaf',10)); lr=float(spec.get('learning_rate',.04))
    if kind=='Ridge':
        estimator=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=spec.get('alpha',30)))
    else:
        if kind in ['XGBoost','LightGBM']:
            try:
                if kind=='XGBoost':
                    from xgboost import XGBRegressor
                    reg=XGBRegressor(n_estimators=n,max_depth=depth,learning_rate=lr,random_state=RANDOM_STATE,n_jobs=2)
                else:
                    from lightgbm import LGBMRegressor
                    reg=LGBMRegressor(n_estimators=n,max_depth=depth,learning_rate=lr,min_child_samples=leaf,random_state=RANDOM_STATE,n_jobs=2,verbosity=-1)
            except ImportError:
                logging.warning('%s 不可用，明确回退 HistGradientBoosting',kind)
                kind='HistGradientBoosting'; m.resolved_name=kind+' (fallback)'
        if kind=='RandomForest':
            reg=RandomForestRegressor(n_estimators=n,max_depth=depth,min_samples_leaf=leaf,random_state=RANDOM_STATE,n_jobs=2)
        elif kind=='GradientBoosting':
            reg=GradientBoostingRegressor(n_estimators=n,max_depth=depth,min_samples_leaf=leaf,learning_rate=lr,loss='huber',random_state=RANDOM_STATE)
        elif kind=='HistGradientBoosting':
            reg=HistGradientBoostingRegressor(max_iter=n,max_depth=depth,min_samples_leaf=leaf,learning_rate=lr,early_stopping=False,random_state=RANDOM_STATE)
        elif kind not in ['XGBoost','LightGBM']:
            raise ValueError(f'不支持模型 {kind}')
        # 禁用会随机抽内部验证集的 early stopping；用外部时间折评估。
        estimator=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),reg)
    estimator.fit(x,y)
    m.estimator=estimator; m.columns=list(x)
    return m

def baseline_value(name,series,date):
    """多步基线同样仅使用起点历史与递归预测。"""
    if name.startswith('Weekly'):
        return float(series.iloc[-7])
    if name=='Yesterday':
        return float(series.iloc[-1])
    if name.startswith('MA'):
        return float(series.iloc[-int(name[2:]):].mean())
    if name=='SameWeekday':
        return float(series[series.index.dayofweek==date.dayofweek].mean())
    raise ValueError(name)

def feature_importance(model):
    """返回模型实际重要性/标准化系数，不伪造 SHAP 或 loss。"""
    if model.columns is None:
        return []
    est=model.estimator.steps[-1][1]
    values=getattr(est,'feature_importances_',None)
    kind='impurity_importance'
    if values is None and hasattr(est,'coef_'):
        values=np.abs(est.coef_); kind='absolute_standardized_coefficient'
    if values is None:
        return []
    return [{'target':model.target,'feature':c,'importance':float(v),'importance_type':kind} for c,v in zip(model.columns,values)]
