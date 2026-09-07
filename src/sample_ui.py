"""多步训练样本审计页：实际使用、消融收益和样本组剔除。"""
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from .multistep_samples import SAMPLE_DIR

def render_sample_page():
    """只展示磁盘上的实际实验记录，不生成假示例或假指标。"""
    st.write('每条训练样本 = 预测起点已知特征 + horizon + 未来标签。界面区分“已构造”“实际入模”“验证有效”三种证据。')
    ledger_path=SAMPLE_DIR/'samples/sample_ledger.csv'
    if not ledger_path.exists():
        st.info('尚无样本账本。运行 python -m src.sample_experiment --stage develop')
        return
    ledger=pd.read_csv(ledger_path,parse_dates=['origin','label_date','feature_observed_max_date'])
    a,b,c=st.columns(3)
    a.metric('扩展样本行',f'{len(ledger):,}');b.metric('真实标签日期',ledger.label_date.nunique());c.metric('历史预测起点',ledger.origin.nunique())
    st.caption('多个起点可能预测同一个日期，所以扩展行数不等于独立样本量。训练按标签日期重复次数校正权重。')
    tabs=st.tabs(['样本账本','实际入模证据','对照实验','有效样本组','冻结与八月审计'])
    with tabs[0]:
        origin=st.selectbox('查看预测起点',ledger.origin.dt.strftime('%Y-%m-%d').unique())
        part=ledger[ledger.origin==pd.Timestamp(origin)]
        st.dataframe(part,hide_index=True)
        x=pd.read_csv(SAMPLE_DIR/'samples/features.csv.gz',index_col='sample_id')
        y=pd.read_csv(SAMPLE_DIR/'samples/normalized_labels.csv.gz',index_col='sample_id')
        scale=pd.read_csv(SAMPLE_DIR/'samples/origin_scales.csv.gz',index_col='sample_id')
        ids=part.sample_id.tolist()
        st.write('该起点实际特征（用于fit的列由所选实验决定）')
        st.dataframe(x.loc[ids])
        st.write('标签还原到原始金额单位')
        st.dataframe(y.loc[ids]*scale.loc[ids,y.columns])
        st.download_button('下载完整样本账本',ledger_path.read_bytes(),'sample_ledger.csv','text/csv')
        st.caption('用户群阈值仅用起点前56天历史。inactive_or_unseen表示近期未观测或尚未出现者，不代表已知的新注册用户。')
    with tabs[1]:
        files=sorted((SAMPLE_DIR/'fit_receipts').rglob('*.json'))
        if files:
            file=st.selectbox('选择一次实际拟合记录',files,format_func=lambda p:str(p.relative_to(SAMPLE_DIR/'fit_receipts')))
            receipts=json.loads(file.read_text(encoding='utf8'))
            for r in receipts:
                st.json({k:v for k,v in r.items() if k not in ['feature_columns','used_split_features','spec']})
                used=pd.Series(r['used_split_features']).sort_values(ascending=False).head(20)
                if len(used):
                    st.plotly_chart(px.bar(x=used.values,y=used.index,orientation='h',title='实际树分裂次数最多的特征（不是因果贡献）'),width='stretch')
                with st.expander('全部输入列与实际分裂特征'):
                    st.json({'input_columns':r['feature_columns'],'used_split_features':r['used_split_features']})
            members=file.with_name(file.stem+'_members.csv.gz')
            if members.exists():
                st.dataframe(pd.read_csv(members).head(100),hide_index=True)
        else:
            st.info('样本已构建，尚未产生实际拟合记录。')
    with tabs[2]:
        path=SAMPLE_DIR/'development_metrics.csv'
        if path.exists():
            metrics=pd.read_csv(path)
            selected=st.multiselect('比较实验',metrics.Experiment.unique(),default=[n for n in ['A_core_weekly','C_features','D_components','H_relative'] if n in metrics.Experiment.unique()])
            st.plotly_chart(px.line(metrics[metrics.Experiment.isin(selected)],x='Period',y='Weighted_Error',color='Experiment',markers=True),width='stretch')
            st.dataframe(metrics,hide_index=True)
        path=SAMPLE_DIR/'ablation_comparison.csv'
        if path.exists():
            st.dataframe(pd.read_csv(path),hide_index=True)
            st.caption('mean_mape_delta_pp为子实验减对照实验的MAPE百分点，负数表示改善；consistent_benefit要求平均改善且至少两个开发月改善。')
    with tabs[3]:
        path=SAMPLE_DIR/'effective_sample_groups.csv'
        if path.exists():
            st.dataframe(pd.read_csv(path),hide_index=True)
            st.caption('删除某组后平均误差上升，并且至少两个验证月恶化，才记录为“该样本组有作用的证据”。这不能证明组内每条记录都有效。')
            st.dataframe(pd.read_csv(SAMPLE_DIR/'sample_group_removal.csv'),hide_index=True)
            examples=SAMPLE_DIR/'effective_sample_examples.csv'
            if examples.exists():
                st.dataframe(pd.read_csv(examples),hide_index=True)
        else:
            st.info('样本组删除实验尚未完成。')
    with tabs[4]:
        path=SAMPLE_DIR/'frozen_selection.json'
        if path.exists():
            frozen=json.loads(path.read_text(encoding='utf8'))
            st.json({'chosen':frozen['chosen'],'selection_end':frozen['selection_end'],'selection_rule':frozen['selection_rule']})
        path=SAMPLE_DIR/'august_comparison.csv'
        if path.exists():
            st.dataframe(pd.read_csv(path),hide_index=True)
            st.caption('八月仅在方案冻结后审计，且此前已被观察；模拟分不代表官方分数。')
            release=json.loads((SAMPLE_DIR/'release_check.json').read_text(encoding='utf8'))
            st.json(release)
            if not release['passed']:
                st.warning('候选未通过整体误差/模拟分验收，原默认提交文件未改动。')
            p=SAMPLE_DIR/'prediction_201409_multistep.csv'
            if p.exists():
                st.download_button('下载多步样本研究候选CSV',p.read_bytes(),p.name,'text/csv')
        st.code('python -m src.sample_experiment --stage develop\npython -m src.sample_experiment --stage audit',language='bash')
