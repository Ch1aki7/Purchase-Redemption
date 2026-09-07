"""仅利用已结束开发折确定不同目标的凸组合权重。"""
import numpy as np
import pandas as pd
from .config import TARGETS
from .metrics import relative_error

def choose_weights(predictions,actual):
    """搜索最佳单模型和两模型0.1步长凸组合，目标分别最小化相对误差。"""
    names=list(predictions); weights={}
    for t in TARGETS:
        y=actual['total_'+t].to_numpy()
        best=(float('inf'),None)
        for a in names:
            for b in names:
                for w in np.linspace(0,1,11):
                    p=w*predictions[a][t].to_numpy()+(1-w)*predictions[b][t].to_numpy()
                    loss=float(relative_error(y,p).mean())
                    if loss<best[0]:
                        candidate={a:float(w),b:float(1-w)} if a!=b else {a:1.0}
                        best=(loss,{k:v for k,v in candidate.items() if v>0})
        weights[t]=best[1]
    return weights

def blend(predictions,weights):
    """按目标融合；检查非负归一权重和索引一致性。"""
    index=next(iter(predictions.values())).index
    out=pd.DataFrame(index=index)
    for t in TARGETS:
        w=weights[t]
        if not np.isclose(sum(w.values()),1) or min(w.values())<0:
            raise ValueError('融合权重必须非负且和为1')
        for k in w:
            if not predictions[k].index.equals(index):
                raise ValueError('融合模型预测日期不一致')
        out[t]=sum(v*predictions[k][t] for k,v in w.items())
    return out
