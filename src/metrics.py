"""真实可计算指标与明确标注的模拟得分。"""
import numpy as np
import pandas as pd
from .config import EPSILON, TARGETS, PURCHASE_WEIGHT, REDEEM_WEIGHT

def relative_error(actual,pred):
    """相对误差以一个原始金额单位为分母下限。"""
    return np.abs(np.asarray(pred)-np.asarray(actual))/np.maximum(np.abs(actual),EPSILON)

def simulated_score(error):
    """线性模拟评分，不代表官方真实评分公式。"""
    return 10*np.maximum(0,1-np.asarray(error)/.3)

def error_frame(actual,pred):
    """严格对齐真实值和预测值，逐日误差与模拟得分。"""
    if not actual.index.equals(pred.index):
        raise ValueError('实际值与预测日期未对齐')
    out=pd.DataFrame(index=pred.index)
    for t in TARGETS:
        a=actual['total_'+t]; p=pred[t]
        out['actual_'+t]=a; out['pred_'+t]=p
        out[t+'_abs_error']=(p-a).abs()
        out[t+'_relative_error']=relative_error(a,p)
        out[t+'_simulated_score']=simulated_score(out[t+'_relative_error'])
    out['daily_simulated_score']=PURCHASE_WEIGHT*out.purchase_simulated_score+REDEEM_WEIGHT*out.redeem_simulated_score
    return out

def evaluate(actual,pred):
    """MAE、RMSE、MAPE(百分比)、加权相对误差及周期模拟分。"""
    e=error_frame(actual,pred); result={}
    for t in TARGETS:
        err=e[t+'_abs_error']; rel=e[t+'_relative_error']; prefix=t.capitalize()
        result.update({prefix+'_MAE':err.mean(),prefix+'_RMSE':np.sqrt((err**2).mean()),prefix+'_MAPE':rel.mean()*100,prefix+'_Relative_Error':rel.mean(),prefix+'_Score':e[t+'_simulated_score'].sum(),prefix+'_Near_Zero_Count':int((actual['total_'+t].abs()<EPSILON).sum())})
    result['Weighted_Error']=PURCHASE_WEIGHT*result['Purchase_Relative_Error']+REDEEM_WEIGHT*result['Redeem_Relative_Error']
    result['Simulated_Score']=e.daily_simulated_score.sum()
    result['Simulated_Score_30day_equivalent']=e.daily_simulated_score.mean()*30
    result['Days']=len(e)
    return result
