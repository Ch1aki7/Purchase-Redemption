"""按月固定起点递归回测，六月七月开发、八月独立检验。"""
import logging
import pandas as pd
from .config import FOLDS, TARGETS, PROCESSED_DATA_DIR, OUTPUT_DIR
from .models import model_specs, fit_model
from .predict import recursive_forecast
from .metrics import evaluate, error_frame
from .ensemble import choose_weights,blend
from .feature_engineering import validate_no_leakage
from .utils import setup_logging,save_json

def run_backtest(d=None,specs=None,save=True):
    """每折重新拟合；融合只用之前已结束折，八月权重不再更新。"""
    setup_logging()
    if d is None:
        d=pd.read_csv(PROCESSED_DATA_DIR/'daily_balance.csv',parse_dates=['date'],index_col='date')
    specs=model_specs() if specs is None else specs
    metrics=[]; all_errors=[]; previous={name:[] for name in specs}; previous_actual=[]; fold_predictions={}; audits=[]
    for fold,(start,end) in enumerate(FOLDS,1):
        train=d.loc[d.index<pd.Timestamp(start)]; actual=d.loc[start:end]
        audits.append(validate_no_leakage(train,actual.index))
        predictions={}
        for name,spec in specs.items():
            logging.info('Fold %s %s 训练截至 %s',fold,name,train.index.max().date())
            fitted={t:fit_model(train,t,spec) for t in TARGETS}
            pred=recursive_forecast(train,actual.index,fitted)
            predictions[name]=pred
            metrics.append({'Model':name,'Fold':fold,'Period':start[:7],'Role':'holdout' if fold==3 else 'development',**evaluate(actual,pred)})
            e=error_frame(actual,pred); e['Model']=name; e['Fold']=fold; all_errors.append(e)
        if previous_actual:
            weights=choose_weights({k:pd.concat(v) for k,v in previous.items()},pd.concat(previous_actual))
        else:
            first='Weekly' if 'Weekly' in specs else next(iter(specs))
            weights={t:{first:1.0} for t in TARGETS}
        ensemble_pred=blend(predictions,weights)
        metrics.append({'Model':'Ensemble_prequential','Fold':fold,'Period':start[:7],'Role':'holdout' if fold==3 else 'development',**evaluate(actual,ensemble_pred)})
        e=error_frame(actual,ensemble_pred);e['Model']='Ensemble_prequential';e['Fold']=fold;all_errors.append(e)
        fold_predictions[fold]=predictions
        if fold<3:
            for name in specs:
                previous[name].append(predictions[name])
            previous_actual.append(actual)
        logging.info('Fold %s 完成，融合权重=%s',fold,weights)
    dev_actual=pd.concat(previous_actual)
    dev_preds={k:pd.concat(v) for k,v in previous.items()}
    final_weights=choose_weights(dev_preds,dev_actual)
    comparison=[]
    for name,pred in dev_preds.items():
        comparison.append({'Model':name,'Role':'development_selection',**evaluate(dev_actual,pred)})
    comparison.append({'Model':'Ensemble_fitted_on_development','Role':'development_selection_optimistic',**evaluate(dev_actual,blend(dev_preds,final_weights))})
    dev=pd.DataFrame(comparison).sort_values(['Weighted_Error','Simulated_Score'],ascending=[True,False])
    mf=pd.DataFrame(metrics); errors=pd.concat(all_errors).rename_axis('date')
    if save:
        mf.to_csv(OUTPUT_DIR/'metrics.csv',index=False)
        errors.to_csv(OUTPUT_DIR/'validation_predictions.csv')
        dev.to_csv(OUTPUT_DIR/'model_comparison.csv',index=False)
        mf[mf.Fold==3].sort_values('Weighted_Error').to_csv(OUTPUT_DIR/'holdout_comparison.csv',index=False)
        errors[(errors.Fold==3)&(errors.Model=='Ensemble_prequential')].to_csv(OUTPUT_DIR/'error_analysis.csv')
        selection={'weights':final_weights,'selection_period':'2014-06-01/2014-07-31','untouched_holdout':'2014-08-01/2014-08-31','specs':specs,'policy':'超参数候选在回测前声明，八月不参与调参或融合权重；最终模型使用截至八月底历史重训。'}
        save_json(selection,OUTPUT_DIR/'selection.json')
        save_json(audits,OUTPUT_DIR/'leakage_audit.json')
        from .eda import error_insights
        held=errors[(errors.Fold==3)&(errors.Model=='Ensemble_prequential')]
        (OUTPUT_DIR/'error_insights.txt').write_text('\n'.join(error_insights(held)),encoding='utf8')
    return mf,errors,dev,final_weights

if __name__=='__main__':
    results=run_backtest()
    print(results[2].to_string(index=False))
