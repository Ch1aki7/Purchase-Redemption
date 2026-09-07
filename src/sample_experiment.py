"""真实多步样本实验：可追踪入模、逐项消融、样本组剔除及冻结后审计。"""
import argparse
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .config import TARGETS,MODEL_DIR,OUTPUT_DIR
from .utils import setup_logging,save_json
from .metrics import evaluate,error_frame,relative_error
from .multistep_samples import SAMPLE_DIR,build_samples,save_bundle,load_inputs,frame_hash
from .traced_direct import experiments,fit_traced,predict_at_origin,select_rows

CUTOFF=pd.Timestamp('2014-07-31')
FOLDS=[('2014-05-01','2014-05-31'),('2014-06-01','2014-06-30'),('2014-07-01','2014-07-31')]
PARENTS={'B_absolute':'A_core_weekly','C_features':'B_absolute','D_components':'C_features','E_cohorts':'C_features','F_dense':'C_features','G_month_end':'F_dense','H_relative':'F_dense','I_component_relative':'H_relative','J_cohort_relative':'H_relative','K_horizon_groups':'H_relative','L_recent120':'H_relative','M_recent60':'H_relative'}

def choose_winners(metrics):
    """只按开发期目标MAPE选型，要求相对基准至少两个验证月改善。"""
    if metrics.Period.max()>'2014-07':
        raise ValueError('选择函数拒绝八月指标')
    chosen={}
    for t in TARGETS:
        col=t.capitalize()+'_MAPE'
        base=metrics[metrics.Experiment=='A_core_weekly'].set_index('Period')[col]
        eligible=[]
        for name,part in metrics.groupby('Experiment'):
            values=part.set_index('Period')[col].reindex(base.index)
            if name=='A_core_weekly' or (values.lt(base).sum()>=2 and values.mean()<base.mean()):
                eligible.append((float(values.mean()),name))
        chosen[t]=min(eligible)[1]
    return chosen

def summarize_ablation(metrics):
    """每个对照只改变明确的一个因素，负delta表示改进。"""
    rows=[]
    for child,parent in PARENTS.items():
        c=metrics[metrics.Experiment==child].set_index('Period')
        p=metrics[metrics.Experiment==parent].set_index('Period')
        for t in TARGETS:
            col=t.capitalize()+'_MAPE';delta=c[col]-p[col]
            rows.append({'experiment':child,'parent':parent,'target':t,'mean_mape_delta_pp':float(delta.mean()),'improved_months':int((delta<0).sum()),'tested_months':len(delta),'consistent_benefit':bool((delta<0).sum()>=2 and delta.mean()<0),'per_month_delta':json.dumps(delta.to_dict())})
    return pd.DataFrame(rows)

def development(daily,raw):
    """入口物理截断至七月底，所有候选预测与选择留在开发月份。"""
    if daily.index.max()>CUTOFF or raw.date.max()>CUTOFF:
        raise ValueError('开发函数不接受八月明细或汇总')
    specs=experiments();SAMPLE_DIR.mkdir(parents=True,exist_ok=True)
    save_json({'candidate_specs':specs,'parents':PARENTS,'folds':FOLDS,'selection_rule':'每目标开发期平均MAPE最小，且至少2/3个月优于A；不按八月选型','sample_counts_warning':'同一标签日可对应多个起点，扩展行数不等于独立真实天数；训练按标签重复数调权','research_status':'八月之前已被观察，后续八月仅冻结后事后审计'},SAMPLE_DIR/'protocol.json')
    bundle,snapshots=build_samples(daily,raw,CUTOFF)
    save_bundle(bundle,snapshots,SAMPLE_DIR/'samples')
    joblib.dump(bundle,SAMPLE_DIR/'development_bundle.joblib')
    metrics=[];errors=[]
    for start,end in FOLDS:
        cutoff=pd.Timestamp(start)-pd.Timedelta(days=1);dates=pd.date_range(start,end)
        actual=daily.loc[dates].rename(columns={'total_purchase':'total_purchase','total_redeem':'total_redeem'})
        for i,(name,spec) in enumerate(specs.items(),1):
            logging.info('TRACE FIT %s %s/%s %s',start[:7],i,len(specs),name)
            models={t:fit_traced(bundle,cutoff,spec,t,name,SAMPLE_DIR/'fit_receipts'/start[:7]) for t in TARGETS}
            p=predict_at_origin(models,daily,raw,cutoff,dates)
            metrics.append({'Experiment':name,'Period':start[:7],**evaluate(actual,p)})
            e=error_frame(actual,p);e['Experiment']=name;e['Period']=start[:7];errors.append(e)
            pd.DataFrame(metrics).to_csv(SAMPLE_DIR/'development_metrics.csv',index=False)
    metrics=pd.DataFrame(metrics);errors=pd.concat(errors)
    errors.to_csv(SAMPLE_DIR/'development_predictions.csv',index_label='date')
    summarize_ablation(metrics).to_csv(SAMPLE_DIR/'ablation_comparison.csv',index=False)
    chosen=choose_winners(metrics)
    # 先冻结，样本删除实验只用于解释，不再根据这些解释调整模型。
    frozen={'chosen':chosen,'specs':specs,'selection_end':str(CUTOFF.date()),'x_hash':frame_hash(bundle.x),'label_hash':frame_hash(bundle.y),'sample_rows':len(bundle.x),'unique_label_dates':bundle.meta.label_date.nunique(),'unique_origins':bundle.meta.origin.nunique(),'august_used_for_selection':False,'selection_rule':'MAPE，至少2个开发月优于A，否则保留A'}
    save_json(frozen,SAMPLE_DIR/'frozen_selection.json')
    logging.info('TRACED MODELS FROZEN: %s',chosen)
    removals=[]
    for start,end in FOLDS:
        cutoff=pd.Timestamp(start)-pd.Timedelta(days=1);dates=pd.date_range(start,end)
        actual=daily.loc[dates]
        for t,name in chosen.items():
            spec=specs[name]
            full_rows=len(select_rows(bundle,cutoff,spec).x)
            reference=metrics[(metrics.Experiment==name)&(metrics.Period==start[:7])].iloc[0][t.capitalize()+'_MAPE']
            for group in ['recent_0_30','middle_31_90','older_91_plus','month_end_origins']:
                kept=len(select_rows(bundle,cutoff,spec,group).x);removed=full_rows-kept
                if kept<40 or removed==0:
                    removals.append({'Period':start[:7],'target':t,'Experiment':name,'removed_group':group,'rows_removed':removed,'status':'not_testable','reference_MAPE':reference})
                    continue
                logging.info('REMOVE SAMPLE GROUP %s %s %s',start[:7],t,group)
                model=fit_traced(bundle,cutoff,spec,t,name,SAMPLE_DIR/'removal_receipts'/start[:7],group)
                p=predict_at_origin({t:model},daily,raw,cutoff,dates)
                error=float(relative_error(actual['total_'+t],p[t]).mean()*100)
                removals.append({'Period':start[:7],'target':t,'Experiment':name,'removed_group':group,'rows_removed':removed,'status':'tested','reference_MAPE':reference,'removed_MAPE':error,'delta_after_removal_pp':error-reference})
    removal=pd.DataFrame(removals);removal.to_csv(SAMPLE_DIR/'sample_group_removal.csv',index=False)
    valid=removal[removal.status=='tested']
    summary=valid.groupby(['target','removed_group']).agg(mean_error_increase_pp=('delta_after_removal_pp','mean'),worse_after_removal_months=('delta_after_removal_pp',lambda s:int((s>0).sum())),tested_months=('Period','count')).reset_index()
    summary['evidence_of_usefulness']=(summary.mean_error_increase_pp>0)&(summary.worse_after_removal_months>=2)
    summary.to_csv(SAMPLE_DIR/'effective_sample_groups.csv',index=False)
    # 汇总实际模型读取了哪些特征，以及树是否实际在该特征上发生分裂。
    receipts=[]
    for path in (SAMPLE_DIR/'fit_receipts').rglob('*.json'):
        for r in json.loads(path.read_text(encoding='utf8')):
            receipts.append({k:v for k,v in r.items() if k not in ['feature_columns','spec','used_split_features']})
    pd.DataFrame(receipts).to_csv(SAMPLE_DIR/'training_usage_summary.csv',index=False)
    return frozen

def audit(daily,raw):
    """只按已冻结的选择审计一次八月，并输出单独的九月研究候选。"""
    frozen=json.loads((SAMPLE_DIR/'frozen_selection.json').read_text(encoding='utf8'))
    if frozen['selection_end']>'2014-07-31':
        raise ValueError('冻结方案使用了八月')
    bundle=joblib.load(SAMPLE_DIR/'development_bundle.joblib')
    if frame_hash(bundle.x)!=frozen['x_hash'] or frame_hash(bundle.y)!=frozen['label_hash']:
        raise ValueError('开发样本已变化，冻结模型无效')
    models={}
    for t,name in frozen['chosen'].items():
        models[t]=fit_traced(bundle,CUTOFF,frozen['specs'][name],t,name,SAMPLE_DIR/'audit_fit_receipts')
    dates=pd.date_range('2014-08-01','2014-08-31')
    pred=predict_at_origin(models,daily.loc[:CUTOFF],raw.loc[raw.date<=CUTOFF],CUTOFF,dates)
    pred.to_csv(SAMPLE_DIR/'august_predictions.csv',index_label='date')
    actual=daily.loc[dates]
    error_frame(actual,pred).to_csv(SAMPLE_DIR/'august_errors.csv',index_label='date')
    metrics=evaluate(actual,pred)
    save_json(metrics,SAMPLE_DIR/'august_metrics.json')
    # 九月重训可使用八月，此时选择规则早已冻结；明细另存避免混淆。
    full_bundle,snapshot=build_samples(daily,raw,pd.Timestamp('2014-08-31'))
    save_bundle(full_bundle,snapshot,SAMPLE_DIR/'final_training_samples')
    save_dir=MODEL_DIR/'multistep';save_dir.mkdir(parents=True,exist_ok=True)
    final_models={}
    for t,name in frozen['chosen'].items():
        final_models[t]=fit_traced(full_bundle,pd.Timestamp('2014-08-31'),frozen['specs'][name],t,name,SAMPLE_DIR/'final_fit_receipts')
        joblib.dump(final_models[t],save_dir/(t+'.joblib'))
    future=predict_at_origin(final_models,daily,raw,pd.Timestamp('2014-08-31'),pd.date_range('2014-09-01','2014-09-30'))
    if len(future)!=30 or not np.isfinite(future).all().all() or (future<0).any().any():
        raise ValueError('预测输出合同失败')
    save_json({'negative_raw_predictions':future.attrs['negative_raw_predictions'],'finite':True,'nonnegative':True,'rows':len(future)},SAMPLE_DIR/'prediction_validation.json')
    future=future.round().astype('int64');future.index=future.index.strftime('%Y%m%d')
    future.to_csv(SAMPLE_DIR/'prediction_201409_multistep.csv',index_label='date')
    future.to_csv(SAMPLE_DIR/'prediction_201409_multistep_no_header.csv',header=False)
    save_json({**frozen,'train_end':'2014-08-31'},save_dir/'manifest.json')
    old=pd.read_csv(OUTPUT_DIR/'holdout_comparison.csv')
    old=old[old.Model.isin(['Ensemble_prequential','RandomForest'])]
    compare=pd.concat([old,pd.DataFrame([{'Model':'Traced_multistep_frozen',**metrics}])],ignore_index=True)
    compare.to_csv(SAMPLE_DIR/'august_comparison.csv',index=False)
    baseline=old[old.Model=='Ensemble_prequential'].iloc[0]
    save_json({'passed':bool(metrics['Weighted_Error']<baseline.Weighted_Error and metrics['Simulated_Score']>=baseline.Simulated_Score),'default_replaced':False,'rule':'只验收冻结候选，整体误差下降且模拟分不下降；本程序不会覆盖原默认文件'},SAMPLE_DIR/'release_check.json')
    return metrics

def main():
    """开发阶段和八月审计阶段单独运行，禁止以审计结果反馈选择。"""
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['develop','audit'],required=True);args=parser.parse_args()
    setup_logging();daily,raw=load_inputs()
    if args.stage=='develop':
        development(daily.loc[:CUTOFF],raw.loc[raw.date<=CUTOFF])
    else:
        print(audit(daily,raw))

if __name__=='__main__':
    main()
