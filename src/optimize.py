"""仅用2014年7月及以前的按月回测选择保留峰谷的模型。"""
import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .config import OUTPUT_DIR,MODEL_DIR,PROCESSED_DATA_DIR,TARGETS
from .models import fit_model
from .predict import recursive_forecast
from .peak_models import fit_calendar,fit_direct,make_direct_training,forecast_new
from .metrics import evaluate,error_frame,relative_error
from .utils import setup_logging,save_json

OPT_DIR=OUTPUT_DIR/'optimization'
OPT_MODELS=MODEL_DIR/'optimization'
DEVELOPMENT_END=pd.Timestamp('2014-07-31')
DEV_FOLDS=[('2014-05-01','2014-05-31'),('2014-06-01','2014-06-30'),('2014-07-01','2014-07-31')]

def candidate_specs():
    """搜索空间事前声明，未编码八月的日期或真实金额。"""
    specs={}
    for window in [90,150,240]:
        for alpha in [.3,3,15]:
            for log in [False,True]:
                name=f'Calendar_w{window}_a{alpha}_log{int(log)}'
                specs[name]={'family':'calendar','window':window,'alpha':alpha,'log':log,'style':'day','half_life':60,'trend':False}
    for window in [120,180]:
        for alpha in [1,10]:
            for trend in [False,True]:
                name=f'Fourier_w{window}_a{alpha}_trend{int(trend)}'
                specs[name]={'family':'calendar','window':window,'alpha':alpha,'log':False,'style':'fourier','half_life':60,'trend':trend}
    for window in [120,210]:
        for leaves,leaf in [(7,30),(15,20),(31,10)]:
            name=f'Direct_w{window}_leaves{leaves}'
            specs[name]={'family':'direct','window':window,'half_life':90,'estimator':'hgb','iterations':160,'leaves':leaves,'leaf':leaf,'l2':3}
        for alpha in [10,100]:
            specs[f'DirectRidge_w{window}_a{alpha}']={'family':'direct','window':window,'half_life':90,'estimator':'ridge','alpha':alpha}
    for window in [120,210,427]:
        for leaf in [2,8]:
            specs[f'RF_w{window}_leaf{leaf}']={'family':'recursive','window':window,'kind':'RandomForest','n_estimators':180,'max_depth':8,'min_samples_leaf':leaf}
    specs['RF_original']={'family':'recursive','window':427,'kind':'RandomForest','n_estimators':180,'max_depth':5,'min_samples_leaf':8}
    return specs

def fit_candidate(history,spec):
    """按方案训练两目标，不允许训练函数看到未来标签。"""
    if spec['family']=='calendar':
        return {t:fit_calendar(history,t,spec) for t in TARGETS}
    if spec['family']=='direct':
        prepared=make_direct_training(history,spec['window'])
        return {t:fit_direct(history,t,spec,prepared) for t in TARGETS}
    return {t:fit_model(history.iloc[-spec['window']:],t,spec) for t in TARGETS}

def predict_candidate(models,history,dates,spec):
    """递归模型与直接多步模型使用同一月起点。"""
    return recursive_forecast(history,dates,models) if spec['family']=='recursive' else forecast_new(models,history,dates)

def shape_metrics(actual,pred,target):
    """峰谷仅在评分时由各折真实值定义，不作为预测输入。"""
    a=actual['total_'+target];p=pred[target];e=relative_error(a,p)
    groups=a.index.to_period('M')
    peak=a>=a.groupby(groups).transform(lambda s:s.quantile(.8))
    valley=a<=a.groupby(groups).transform(lambda s:s.quantile(.2))
    delta=(p.groupby(groups).diff()-a.groupby(groups).diff()).abs().mean()/max(a.mean(),1)
    peak_error=float(e[peak].mean());valley_error=float(e[valley].mean())
    objective=.60*float(e.mean())+.15*peak_error+.15*valley_error+.10*float(delta)
    top_true=set(a.nlargest(max(1,len(a)//5)).index)
    top_pred=set(p.nlargest(max(1,len(p)//5)).index)
    return {'MAPE':float(e.mean()),'Peak_MAPE':peak_error,'Valley_MAPE':valley_error,'Delta_Error':float(delta),'Shape_Objective':objective,'Amplitude_Ratio':float(p.std()/max(a.std(),1)),'Top20_Recall':len(top_true&top_pred)/len(top_true)}

def select_on_development(actual,preds):
    """每个目标独立选两模型凸组合，只接收开发期真值与预测。"""
    if actual.index.max()>DEVELOPMENT_END:
        raise ValueError('参数选择禁止读取八月标签')
    weights={};rows=[]
    for t in TARGETS:
        ranked=[]
        for name,p in preds.items():
            metric=shape_metrics(actual,p,t)
            rows.append({'Model':name,'Target':t,**metric})
            ranked.append((metric['Shape_Objective'],name))
        # 仅在开发期排名前8候选内搜索融合，限制搜索容量。
        names=[name for _,name in sorted(ranked)[:8]]
        best=(float('inf'),None)
        for i,a in enumerate(names):
            for b in names[i:]:
                for w in [0,.25,.5,.75,1]:
                    p=preds[a]*w+preds[b]*(1-w)
                    loss=shape_metrics(actual,p,t)['Shape_Objective']
                    if loss<best[0]:
                        ws={a:1.0} if a==b else {a:w,b:1-w}
                        best=(loss,{k:v for k,v in ws.items() if v>0})
        weights[t]=best[1]
    return weights,pd.DataFrame(rows)

def blend_selected(preds,weights):
    """两个目标可选择不同家族和不同权重。"""
    return pd.DataFrame({t:sum(w*preds[n][t] for n,w in weights[t].items()) for t in TARGETS})

def optimize(development):
    """训练/调参入口在结构上只能访问八月以前的数据。"""
    if development.index.max()>DEVELOPMENT_END:
        raise ValueError('开发数据必须截至2014-07-31')
    setup_logging();OPT_DIR.mkdir(exist_ok=True);OPT_MODELS.mkdir(exist_ok=True)
    specs=candidate_specs()
    protocol={'development_end':str(DEVELOPMENT_END.date()),'folds':DEV_FOLDS,'objective':'.60 MAPE + .15 peak20 MAPE + .15 valley20 MAPE + .10 monthly difference error / mean actual','candidate_specs':specs,'august_policy':'八月已在上一轮被观察，仅作冻结后事后审计，不用于本次拟合或选择；不保证指定日期可预测。'}
    save_json(protocol,OPT_DIR/'protocol.json')
    per_model={n:[] for n in specs};actuals=[];metrics=[]
    for start,end in DEV_FOLDS:
        train=development.loc[development.index<pd.Timestamp(start)]
        actual=development.loc[start:end]
        for i,(name,spec) in enumerate(specs.items(),1):
            logging.info('PRE-AUGUST %s %s/%s %s',start[:7],i,len(specs),name)
            models=fit_candidate(train,spec)
            pred=predict_candidate(models,train,actual.index,spec)
            per_model[name].append(pred)
            row={'Model':name,'Period':start[:7],**evaluate(actual,pred)}
            for t in TARGETS:
                row.update({t+'_'+k:v for k,v in shape_metrics(actual,pred,t).items()})
            metrics.append(row)
        actuals.append(actual)
        pd.DataFrame(metrics).to_csv(OPT_DIR/'development_metrics.csv',index=False)
    actual=pd.concat(actuals);preds={n:pd.concat(parts) for n,parts in per_model.items()}
    weights,shapes=select_on_development(actual,preds)
    selected=blend_selected(preds,weights)
    final={**protocol,'weights':weights,'selection_max_label_date':str(actual.index.max().date()),'candidate_count':len(specs),'selection_hash':hashlib.sha256(json.dumps(weights,sort_keys=True).encode()).hexdigest(),'development_metrics':evaluate(actual,selected)}
    # 此文件先写出，再允许后续命令进入八月审计。
    save_json(final,OPT_DIR/'frozen_selection.json')
    shapes.to_csv(OPT_DIR/'development_shape_metrics.csv',index=False)
    error_frame(actual,selected).to_csv(OPT_DIR/'development_selected_errors.csv',index_label='date')
    logging.info('FROZEN BEFORE AUGUST AUDIT: %s',weights)
    return final

def audit_and_forecast(full):
    """仅加载冻结方案后审计八月；不会根据八月指标改选或反调。"""
    selection=json.loads((OPT_DIR/'frozen_selection.json').read_text(encoding='utf8'))
    if selection['selection_max_label_date']>'2014-07-31':
        raise ValueError('冻结方案越过开发边界')
    weights=selection['weights'];specs=selection['candidate_specs']
    active=sorted({name for ws in weights.values() for name in ws})
    history=full.loc[:DEVELOPMENT_END];dates=pd.date_range('2014-08-01','2014-08-31')
    predictions={}
    for name in active:
        models=fit_candidate(history,specs[name])
        predictions[name]=predict_candidate(models,history,dates,specs[name])
    pred=blend_selected(predictions,weights)
    # 先保存预测，再读取本函数中的八月真值进行评分。
    pred.to_csv(OPT_DIR/'august_predictions.csv',index_label='date')
    actual=full.loc[dates]
    e=error_frame(actual,pred)
    e.to_csv(OPT_DIR/'august_errors.csv',index_label='date')
    old=pd.read_csv(OUTPUT_DIR/'validation_predictions.csv',parse_dates=['date']).set_index('date')
    rows=[];specific=[]
    for name in ['Ensemble_prequential','RandomForest','Optimized_frozen']:
        if name=='Optimized_frozen':
            p=pred
        else:
            part=old[(old.Model==name)&(old.Fold==3)].sort_index()
            p=part[['pred_purchase','pred_redeem']].rename(columns={'pred_purchase':'purchase','pred_redeem':'redeem'})
        row={'Model':name,**evaluate(actual,p)}
        for t in TARGETS:
            row.update({t+'_'+k:v for k,v in shape_metrics(actual,p,t).items()})
        rows.append(row)
        diagnostic=error_frame(actual,p).loc[pd.to_datetime(['2014-08-05','2014-08-11','2014-08-27'])]
        diagnostic['Model']=name;specific.append(diagnostic)
    comparison=pd.DataFrame(rows)
    comparison.to_csv(OPT_DIR/'august_comparison.csv',index=False)
    old_metrics=comparison.set_index('Model').loc['Ensemble_prequential']
    new_metrics=comparison.set_index('Model').loc['Optimized_frozen']
    release={'approved_for_default_submission':bool(new_metrics.Weighted_Error < old_metrics.Weighted_Error and new_metrics.Simulated_Score >= old_metrics.Simulated_Score),'baseline_weighted_error':float(old_metrics.Weighted_Error),'candidate_weighted_error':float(new_metrics.Weighted_Error),'baseline_simulated_score':float(old_metrics.Simulated_Score),'candidate_simulated_score':float(new_metrics.Simulated_Score),'rule':'整体加权误差下降且模拟分不下降才通过替换验收；只验收冻结候选，不依八月另选模型。','default_prediction_replaced':False,'selection_hash':selection['selection_hash']}
    save_json(release,OPT_DIR/'release_check.json')
    pd.concat(specific).to_csv(OPT_DIR/'requested_dates_diagnostic.csv',index_label='date')
    future_dates=pd.date_range('2014-09-01','2014-09-30');future={};negative={}
    for name in active:
        models=fit_candidate(full,specs[name])
        for t,m in models.items():
            joblib.dump(m,OPT_MODELS/(t+'_'+name+'.joblib'))
        future[name]=predict_candidate(models,full,future_dates,specs[name])
        negative[name]=future[name].attrs.get('negative_predictions',{})
    result=blend_selected(future,weights).round().astype('int64')
    if len(result)!=30 or not np.isfinite(result).all().all() or (result<0).any().any():
        raise ValueError('九月预测合同失败')
    result.index=result.index.strftime('%Y%m%d')
    result.to_csv(OPT_DIR/'prediction_201409_optimized.csv',index_label='date')
    result.to_csv(OPT_DIR/'prediction_201409_optimized_no_header.csv',header=False)
    save_json({**selection,'active_models':active,'final_train_end':str(full.index.max().date()),'negative_raw_predictions':negative,'august_used_for_selection':False},OPT_MODELS/'manifest.json')
    return pd.DataFrame(rows)

def main():
    """开发与审计可分别调用，便于检查冻结方案后才接触八月结果。"""
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['develop','audit','all'],default='all');args=parser.parse_args()
    setup_logging()
    full=pd.read_csv(PROCESSED_DATA_DIR/'daily_balance.csv',parse_dates=['date'],index_col='date')
    if args.stage in ['develop','all']:
        optimize(full.loc[:DEVELOPMENT_END].copy())
    if args.stage in ['audit','all']:
        print(audit_and_forecast(full).to_string(index=False))

if __name__=='__main__':
    main()
