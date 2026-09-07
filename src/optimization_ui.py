"""峰谷优化实验页：开发期选择与八月事后审计明确分开。"""
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from .optimize import OPT_DIR,DEVELOPMENT_END,optimize,audit_and_forecast
from .config import OUTPUT_DIR

def render_optimization_page(d):
    """展示已冻结的参数、真实峰谷诊断与独立保存的九月候选。"""
    st.write('近期日历 / 直接多步 / 递归树模型。参数只用五月、六月、七月回测选择。八月仅在冻结后审计。')
    st.warning('八月已用于发现旧方案的问题，所以这里标记为事后审计，不再声称是完全未接触的测试集。')
    with st.expander('重新运行实验（耗时数分钟）'):
        st.caption('每次只在相同的七月底以前开发数据上搜索；不要依据八月分数反复改搜索空间。')
        if st.button('1. 运行开发期搜索并冻结'):
            with st.spinner('43个候选 × 3个月；仅使用2014-07-31以前数据…'):
                optimize(d.loc[:DEVELOPMENT_END].copy())
            st.success('方案已冻结。')
        if st.button('2. 按冻结方案审计八月并预测九月',disabled=not (OPT_DIR/'frozen_selection.json').exists()):
            with st.spinner('审计不会修改已冻结的权重…'):
                audit_and_forecast(d)
            st.success('八月审计和九月预测完成，原提交文件保持不变。')
    selected=OPT_DIR/'frozen_selection.json'
    if not selected.exists():
        st.info('开发期搜索尚未完成。命令：python -m src.optimize --stage develop')
        return
    selection=json.loads(selected.read_text(encoding='utf8'))
    st.subheader('开发期冻结方案')
    st.json({'weights':selection['weights'],'selection_max_label_date':selection['selection_max_label_date'],'objective':selection['objective'],'selection_hash':selection['selection_hash']})
    for target,weights in selection['weights'].items():
        st.write(target)
        for model,weight in weights.items():
            st.json({'model':model,'weight':weight,**selection['candidate_specs'][model]})
    dev=pd.read_csv(OPT_DIR/'development_metrics.csv')
    chosen={name for w in selection['weights'].values() for name in w}
    view=dev[dev.Model.isin(chosen|{'RF_original'})]
    st.plotly_chart(px.line(view,x='Period',y='Weighted_Error',color='Model',markers=True,title='仅五月至七月：选中候选的月度误差'),width='stretch')
    with st.expander('全部开发期搜索记录'):
        st.dataframe(dev,hide_index=True)
        st.dataframe(pd.read_csv(OPT_DIR/'development_shape_metrics.csv'),hide_index=True)
    if not (OPT_DIR/'august_comparison.csv').exists() or (OPT_DIR/'august_comparison.csv').stat().st_mtime_ns < selected.stat().st_mtime_ns:
        st.info('尚未进行冻结后的八月审计。');return
    comparison=pd.read_csv(OPT_DIR/'august_comparison.csv')
    release_path=OPT_DIR/'release_check.json'
    if release_path.exists():
        release=json.loads(release_path.read_text(encoding='utf8'))
        if not release['approved_for_default_submission']:
            st.error('本轮峰谷实验未通过整体改进验收：峰值误差虽降低，整体加权相对误差却上升。原默认提交不变，下载文件仅为实验候选。')
    st.subheader('八月事后审计：旧方案与冻结实验方案')
    st.dataframe(comparison,hide_index=True)
    st.caption('所有分数为线性模拟评分，不是官方真实分。MAPE列为百分比；purchase/redeem_Peak_MAPE等形态诊断列为小数。Amplitude_Ratio是预测/真实标准差比，并非越高越好。')
    new=pd.read_csv(OPT_DIR/'august_errors.csv',parse_dates=['date']).set_index('date')
    old=pd.read_csv(OUTPUT_DIR/'error_analysis.csv',parse_dates=['date']).set_index('date')
    target=st.selectbox('峰谷对比目标',['purchase','redeem'],key='peak_target')
    chart=pd.DataFrame({'实际':new['actual_'+target],'原融合':old['pred_'+target],'优化冻结':new['pred_'+target]})
    st.plotly_chart(px.line(chart,x=chart.index,y=list(chart),markers=True,title='实际值与优化前后预测'),width='stretch')
    st.subheader('用户指出日期的诊断（仅评价，不参与搜索）')
    st.dataframe(pd.read_csv(OPT_DIR/'requested_dates_diagnostic.csv'),hide_index=True)
    p=OPT_DIR/'prediction_201409_optimized.csv'
    if p.exists():
        st.download_button('下载实验候选九月CSV（请先查看验收结果）',p.read_bytes(),p.name,'text/csv')
