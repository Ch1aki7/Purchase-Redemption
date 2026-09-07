"""严格起点多步递归预测和最终提交导出。"""
import json
import logging
import joblib
import numpy as np
import pandas as pd
from .config import PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR, FORECAST_START, FORECAST_END, TARGETS
from .feature_engineering import future_row, validate_no_leakage
from .models import baseline_value, SCALE
from .utils import setup_logging, save_json

def recursive_forecast(history,dates,models):
    """不接收未来真值；两目标各自建模、同步写回后推进下一天。"""
    dates=pd.DatetimeIndex(dates)
    validate_no_leakage(history,dates)
    expected=pd.date_range(history.index.max()+pd.Timedelta(days=1),periods=len(dates))
    if not dates.equals(expected):
        raise ValueError('预测日期必须从历史下一天连续开始')
    state=history.copy(); rows=[]; negatives={t:0 for t in TARGETS}
    ts={t:np.asarray(m.estimator.forecast(len(dates)))*SCALE for t,m in models.items() if m.name=='TimeSeries'}
    for step,date in enumerate(dates):
        values={}
        for t,m in models.items():
            if t in ts:
                value=float(ts[t][step])
            elif m.columns is None:
                value=baseline_value(m.name,state['total_'+t],date)
            else:
                value=float(m.estimator.predict(future_row(date,state,m.columns))[0])*SCALE
            if not np.isfinite(value):
                raise ValueError(f'{date} {t} 模型输出非有限值')
            negatives[t]+=int(value<0)
            values[t]=max(0,value)
        # 同日另一目标也不可使用真值。未训练的目标以周基线递归，供跨目标 lag 使用。
        for t in TARGETS:
            if t not in values:
                values[t]=max(0,baseline_value('Weekly',state['total_'+t],date))
        next_row=state.iloc[-1].copy()
        for t in TARGETS:
            next_row['total_'+t]=values[t]
        state.loc[date]=next_row
        rows.append(values)
    logging.info('递归预测 %s 至 %s 原始负预测并截断计数=%s',dates.min().date(),dates.max().date(),negatives)
    out=pd.DataFrame(rows,index=dates); out.attrs['negative_predictions']=negatives
    return out

def predict_final():
    """使用保存的模型和开发期冻结权重生成30天整数结果。"""
    setup_logging()
    manifest_path=MODEL_DIR/'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('尚未训练，请先运行 python -m src.train')
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    d=pd.read_csv(PROCESSED_DATA_DIR/'daily_balance.csv',parse_dates=['date'],index_col='date')
    dates=pd.date_range(FORECAST_START,FORECAST_END)
    from .ensemble import blend
    preds={}; negative_counts={}
    for name in manifest['active_models']:
        fitted={t:joblib.load(MODEL_DIR/t/(name+'.joblib')) for t in TARGETS}
        preds[name]=recursive_forecast(d,dates,fitted)
        negative_counts[name]=preds[name].attrs['negative_predictions']
    pred=blend(preds,manifest['weights'])
    if len(pred)!=30 or not np.isfinite(pred.to_numpy()).all() or (pred<0).any().any():
        raise ValueError('最终预测格式或数值检查失败')
    out=pred.round().astype('int64'); out.index=out.index.strftime('%Y%m%d'); out.index.name='date'
    out.to_csv(OUTPUT_DIR/'prediction_201409.csv')
    out.to_csv(OUTPUT_DIR/'prediction_201409_no_header.csv',header=False)
    save_json({'rows':len(out),'negative_raw_predictions':negative_counts,'finite':True,'nonnegative':True,'weights':manifest['weights'],'source':'真实历史训练模型递归预测；非样例值'},OUTPUT_DIR/'prediction_validation.json')
    return out

if __name__=='__main__':
    print(predict_final().to_string())
