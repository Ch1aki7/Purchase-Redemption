"""从开发折选型并在全历史重训，分别保存申购/赎回模型。"""
import json
import logging
import joblib
import pandas as pd
from .config import PROCESSED_DATA_DIR, OUTPUT_DIR, MODEL_DIR, TARGETS
from .models import fit_model,feature_importance
from .utils import setup_logging,save_json
from .feature_engineering import make_features, validate_no_leakage

def train_final():
    """不存在开发验证结果时先回测；绝不从八月指标挑模型。"""
    setup_logging()
    if not (OUTPUT_DIR/'selection.json').exists():
        from .backtest import run_backtest
        run_backtest()
    selection=json.loads((OUTPUT_DIR/'selection.json').read_text(encoding='utf8'))
    d=pd.read_csv(PROCESSED_DATA_DIR/'daily_balance.csv',parse_dates=['date'],index_col='date')
    validate_no_leakage(d,features=make_features(d,True),enriched=True)
    active=sorted(set(k for w in selection['weights'].values() for k in w))
    importance=[]
    for name in active:
        for t in TARGETS:
            m=fit_model(d,t,selection['specs'][name])
            joblib.dump(m,MODEL_DIR/t/(name+'.joblib'))
            for row in feature_importance(m):
                importance.append({'Model':name,**row})
    # 另存一个树模型用于解释，明确它不一定是最终入选模型。
    for t in TARGETS:
        name='RandomForest_enriched'
        if name in selection['specs'] and name not in active:
            m=fit_model(d,t,selection['specs'][name])
            for row in feature_importance(m):
                importance.append({'Model':name+' (diagnostic)',**row})
    pd.DataFrame(importance,columns=['Model','target','feature','importance','importance_type']).to_csv(OUTPUT_DIR/'feature_importance.csv',index=False)
    save_json({**selection,'active_models':active,'train_end':str(d.index.max()),'negative_clip':'逐模型逐目标计数见预测报告'},MODEL_DIR/'manifest.json')
    logging.info('最终模型已保存: %s',active)
    return selection

if __name__=='__main__':
    train_final()
