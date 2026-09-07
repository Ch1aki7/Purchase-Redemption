"""可复用的探索图表和由真实计算产生的误差结论。"""
import pandas as pd
import plotly.express as px
from .config import PROCESSED_DATA_DIR, OUTPUT_DIR
from .utils import setup_logging

def eda_figures(d):
    """所有图表源于传入日期范围内的真实数据。"""
    targets=['total_purchase','total_redeem']
    figures={'申购赎回趋势':px.line(d,x=d.index,y=targets), '净资金流':px.line(d,x=d.index,y='net_flow')}
    figures['周几效应']=px.bar(d.groupby(d.index.dayofweek)[targets].mean(),barmode='group',labels={'date':'星期（0=周一）'})
    figures['月内日期效应']=px.line(d.groupby(d.index.day)[targets].mean())
    figures['月度资金趋势']=px.bar(d[targets].resample('MS').sum(),barmode='group')
    figures['申购赎回相关性']=px.scatter(d,x='total_purchase',y='total_redeem')
    for title, cols in [('收益率趋势',[c for c in d if c.startswith('yield')]),('SHIBOR 趋势',[c for c in d if c.startswith('shibor')])]:
        if cols:
            figures[title]=px.line(d,x=d.index,y=cols)
    corr=d.select_dtypes('number').corr()
    figures['相关性热力图']=px.imshow(corr,zmin=-1,zmax=1,color_continuous_scale='RdBu_r')
    for title,fig in figures.items():
        fig.update_layout(title=title,template='plotly_white',legend_title_text='',hovermode='x unified')
    return figures

def error_insights(e):
    """对周末、月末和高峰的结论由实际误差计算，避免硬编码诊断。"""
    lines=[]
    for t in ['purchase','redeem']:
        rel=e[t+'_relative_error']; weekend=e.index.dayofweek>=5
        if weekend.any() and (~weekend).any():
            lines.append(f'{t} 周末平均相对误差 {rel[weekend].mean():.1%}，工作日 {rel[~weekend].mean():.1%}。')
        end=e.index.day>=26
        if end.any():
            bias=(e['pred_'+t]-e['actual_'+t])[end].mean()
            lines.append(f'{t} 月末（26日起）平均偏差 {bias:,.0f} 原始金额单位，表现为'+('低估。' if bias<0 else '高估。'))
        peak=e['actual_'+t]>=e['actual_'+t].quantile(.9)
        bias=(e['pred_'+t]-e['actual_'+t])[peak].mean()
        lines.append(f'{t} 本评价区间最高10%资金流日期平均偏差 {bias:,.0f}；峰值误差属于事后诊断。')
    return lines

if __name__=='__main__':
    setup_logging()
    d=pd.read_csv(PROCESSED_DATA_DIR/'daily_balance.csv',parse_dates=['date'],index_col='date')
    for name,fig in eda_figures(d).items():
        fig.write_html(OUTPUT_DIR/(name+'.html'),include_plotlyjs=True)
    print('Phase 4 EDA 完成：离线交互 HTML 已保存')
