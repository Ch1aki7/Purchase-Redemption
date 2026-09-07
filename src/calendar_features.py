"""确定性日历特征；未提供可靠历史节假日日历，故不虚构节假日标签。"""
import numpy as np
import pandas as pd

def calendar_features(index):
    """仅使用预测日前即可获知的公历特征。"""
    i=pd.DatetimeIndex(index)
    d=pd.DataFrame(index=i)
    for name, values in {'year':i.year,'month':i.month,'day':i.day,'day_of_week':i.dayofweek,'day_of_month':i.day,'week_of_year':i.isocalendar().week.to_numpy(dtype=int),'is_weekend':i.dayofweek>=5,'is_month_start':i.is_month_start,'is_month_end':i.is_month_end,'is_quarter_start':i.is_quarter_start,'is_quarter_end':i.is_quarter_end,'days_to_month_end':i.days_in_month-i.day,'days_from_month_start':i.day-1}.items():
        d[name]=np.asarray(values,dtype=int)
    for name,values,period in [('day_of_week',i.dayofweek,7),('day_of_month',i.day-1,i.days_in_month),('month',i.month-1,12)]:
        d['sin_'+name]=np.sin(2*np.pi*values/period)
        d['cos_'+name]=np.cos(2*np.pi*values/period)
    return d
