"""资金流入流出预测系统：真实数据、严格时序回测与可追溯预测。"""
import json
import os
import subprocess
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from src.config import (
    ROOT, PROCESSED_DATA_DIR, OUTPUT_DIR, MODEL_DIR, TARGETS, FOLDS,
    EVENT_DISTANCE_MANIFEST_PATH, EVENT_DISTANCE_PURCHASE_ONLY_PATH,
    EVENT_DISTANCE_REDEEM_ONLY_PATH, C3_EVENT_DISTANCE_VALIDATION_PATH,
)
from src.eda import eda_figures,error_insights
from src.feature_engineering import make_features
from src.models import fit_model,BASELINES,feature_importance
from src.predict import recursive_forecast
from src.metrics import evaluate,error_frame

st.set_page_config(page_title='资金流入流出预测系统',page_icon='📊',layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1450px} [data-testid="stMetric"]{background:#f1f5f9;border-radius:10px;padding:16px} h1{color:#102a43} .stCaption{color:#486581}</style>''',unsafe_allow_html=True)

@st.cache_data
def read_csv(path,mtime):
    """按文件更新时间失效，避免重跑后显示旧实验结果。"""
    return pd.read_csv(path)

@st.cache_data
def read_submission(path,mtime):
    """读取竞赛要求的无表头三列表，mtime 用于训练后自动刷新。"""
    return pd.read_csv(path,header=None,names=['date','purchase','redeem'])

def table(name,processed=False):
    """从预计算结果读取表；缺文件给出可执行指引。"""
    path=(PROCESSED_DATA_DIR if processed else OUTPUT_DIR)/name
    if not path.exists():
        st.warning(f'缺少 {name}，请在项目目录运行 python run_all.py')
        st.stop()
    return read_csv(str(path),path.stat().st_mtime_ns)

def chart(fig):
    """一致的金融分析图表样式。"""
    fig.update_layout(template='plotly_white',legend_title_text='',margin=dict(l=20,r=20,t=45,b=20))
    st.plotly_chart(fig,width='stretch')

def score_note():
    """在展示模拟分的页面就近标注。"""
    st.caption('模拟评分＝10 × max(0, 1－相对误差/0.3)，不代表官方真实评分。MAPE 按百分比显示；Weighted Error 为小数。31天原始总分与30天等效分分开列出。')

def metrics_cards(m):
    """分别显示两个目标和加权误差。"""
    cols=st.columns(4)
    for c,label,value in zip(cols,['申购 MAPE','赎回 MAPE','加权相对误差','30天等效模拟分'],[f"{m['Purchase_MAPE']:.2f}%",f"{m['Redeem_MAPE']:.2f}%",f"{m['Weighted_Error']:.2%}",f"{m['Simulated_Score_30day_equivalent']:.2f}"]):
        c.metric(label,value)

def flatten_counts(mapping,prefix=()):
    """将嵌套计数字典整理为适合前端展示的行。"""
    rows=[]
    for key,value in mapping.items():
        path=(*prefix,str(key))
        if isinstance(value,dict):
            rows.extend(flatten_counts(value,path))
        else:
            rows.append({'项目':' / '.join(path),'数量':value})
    return rows

def weight_table(weights):
    """将按目标保存的融合权重转成长表。"""
    labels={'purchase':'申购','redeem':'赎回'}
    return pd.DataFrame([
        {'目标':labels.get(target,target),'模型':model,'权重':weight}
        for target,models in weights.items() for model,weight in models.items()
    ])

st.sidebar.title('资金流预测 · 分析台')
page=st.sidebar.radio('功能导航',['首页 / 项目概览','数据探索','特征分析','模型训练','模型评估','滚动回测','未来预测','误差分析'])
run_all_clicked=st.sidebar.button(
    '▶ 运行 run_all.py',key='run_all_shortcut',width='stretch',
    help='重新执行预处理、特征、回测、训练和未来预测',
)
st.sidebar.divider()
st.sidebar.caption('历史：2013.07.01 — 2014.08.31\n\n目标：2014.09.01 — 2014.09.30')
st.sidebar.info('所有金额保留原始数据单位。指导书未说明单位换算，系统不擅自换算为元。')

st.title(page)
if run_all_clicked:
    with st.status('正在运行完整数据与模型流程…',expanded=True) as run_status:
        run_status.write('执行预处理 → 特征校验 → 回测 → 正式训练 → 九月预测')
        process=subprocess.run(
            [sys.executable,str(ROOT/'run_all.py')],cwd=ROOT,
            capture_output=True,text=True,encoding='utf-8',errors='replace',
            env={**os.environ,'PYTHONIOENCODING':'utf-8'},
        )
        if process.returncode:
            run_status.update(label='run_all.py 运行失败',state='error',expanded=True)
            if process.stdout.strip():
                st.code(process.stdout,language='text')
            if process.stderr.strip():
                st.code(process.stderr,language='text')
            st.error(f'完整流程退出码：{process.returncode}')
            st.stop()
        run_status.update(label='run_all.py 已运行完成',state='complete',expanded=False)
    read_csv.clear();read_submission.clear()
    st.success('数据、模型和九月预测已重新生成。')
    if process.stdout.strip():
        with st.expander('查看 run_all.py 输出'):
            st.code(process.stdout,language='text')

d=table('daily_balance.csv',True); d['date']=pd.to_datetime(d.date); d=d.set_index('date')

if page=='首页 / 项目概览':
    st.markdown('### 资金流入流出预测系统')
    st.write('将用户交易、静态画像、收益率与 SHIBOR 转化为每日资金流，比较多类模型，并预测未来30天申购与赎回。')
    a,b,c,e=st.columns(4)
    audit=json.loads((OUTPUT_DIR/'data_quality_report.json').read_text(encoding='utf8'))
    a.metric('历史天数',len(d));b.metric('原始交易记录',f"{audit['user_balance']['shape'][0]:,}");c.metric('预测期限','30 天');e.metric('赎回评分权重','55%')
    chart(px.line(d,x=d.index,y=['total_purchase','total_redeem'],title='历史申购与赎回'))
    st.markdown('**实验边界**：六月、七月用于开发选型和融合；八月用于独立检验。最终重训可使用八月历史，但不会根据八月结果重新调权重。')
    held=table('holdout_comparison.csv'); m=held[held.Model=='Ensemble_prequential'].iloc[0]
    st.subheader('八月独立检验 · 开发期冻结融合')
    metrics_cards(m);score_note()
    st.markdown('**答辩演示路线**：数据质量 → 日历与滞后特征 → 三折回测 → 八月误差诊断 → 九月预测导出。')
    st.warning('九月真实标签未提供，因此没有九月真实误差或官方得分。')

elif page=='数据探索':
    dates=st.sidebar.date_input('EDA 显示范围',value=(d.index.min().date(),d.index.max().date()),min_value=d.index.min().date(),max_value=d.index.max().date())
    if len(dates)!=2:
        st.info('请选择完整的起止日期');st.stop()
    subset=d.loc[str(dates[0]):str(dates[1])]
    if subset.empty:
        st.warning('该区间没有数据');st.stop()
    tabs=st.tabs(['资金流与周期','市场利率','数据质量','用户画像'])
    figs=eda_figures(subset)
    with tabs[0]:
        for name in ['申购赎回趋势','净资金流','周几效应','月内日期效应','月度资金趋势','申购赎回相关性']:
            chart(figs[name])
    with tabs[1]:
        st.caption('图中是向前填充后的历史观测；用于预测时再滞后一天。收益率不同字段口径不同，按原字段分别查看。')
        for name in ['收益率趋势','SHIBOR 趋势']:
            if name in figs:chart(figs[name])
    with tabs[2]:
        report=json.loads((OUTPUT_DIR/'data_quality_report.json').read_text(encoding='utf8'))
        key=st.selectbox('原始数据表',list(report))
        r=report[key]
        shape=r.get('shape',['—','—'])
        quality_cards=st.columns(4)
        quality_cards[0].metric('文件',r['file'])
        quality_cards[1].metric('行数',f'{shape[0]:,}' if isinstance(shape[0],int) else shape[0])
        quality_cards[2].metric('列数',shape[1])
        quality_cards[3].metric('完全重复行',f"{r.get('duplicates',0):,}")
        if r.get('date_range'):
            st.caption(f"日期范围：{r['date_range'][0][:10]} — {r['date_range'][1][:10]}")
        mapping=r.get('mapping')
        if mapping:
            st.markdown('#### 字段标准化映射')
            st.dataframe(pd.DataFrame([
                {'原始字段':source,'标准字段':target} for source,target in mapping.items()
            ]),hide_index=True,width='stretch')
        elif r.get('role'):
            st.info(r['role'])
        if 'missing_rate' in r:
            chart(px.bar(x=list(r['missing_rate']),y=list(r['missing_rate'].values()),labels={'x':'字段','y':'缺失比例'}))
            with st.expander('字段类型、前五行及基础统计'):
                st.markdown('**字段类型**')
                st.dataframe(pd.DataFrame([
                    {'字段':field,'数据类型':dtype} for field,dtype in r['dtypes'].items()
                ]),hide_index=True,width='stretch')
                st.markdown('**前五行**')
                st.dataframe(pd.DataFrame(r['head']),hide_index=True,width='stretch')
                st.markdown('**基础统计**')
                statistics=pd.DataFrame.from_dict(r['statistics'],orient='index').replace('NaN',np.nan)
                statistics.index.name='字段'
                st.dataframe(statistics.reset_index(),hide_index=True,width='stretch')
        st.dataframe(table('validation_report.csv',True),hide_index=True)
        st.subheader('余额一致性')
        consistency=json.loads((OUTPUT_DIR/'balance_consistency.json').read_text(encoding='utf8'))
        check_labels={
            'balance_error':'当日余额等式',
            'previous_record_error':'上一条记录余额等式',
            'previous_balance_link_error':'相邻记录余额衔接',
        }
        consistency_rows=[]
        for check_key,label in check_labels.items():
            values=consistency[check_key]
            consistency_rows.append({
                '检查项目':label,
                '检查记录数':f"{values['checked']:,}",
                '异常数':f"{values['anomalies']:,}",
                '异常率':f"{values['ratio']:.8%}",
                '最大绝对误差':f"{values['max_absolute_error']:,.2f}",
            })
        st.dataframe(pd.DataFrame(consistency_rows),hide_index=True,width='stretch')
        st.metric('首次出现或日期不连续记录',f"{consistency['nonconsecutive_or_first_records']:,}")
        component_labels={
            'purchase_components':'申购组成项',
            'direct_components':'直接申购组成项',
            'redeem_components':'赎回组成项',
            'transfer_components':'转出组成项',
        }
        component_rows=[]
        for component_key,label in component_labels.items():
            values=consistency[component_key]
            component_rows.append({
                '组成项检查':label,
                '异常数':f"{values['anomalies']:,}",
                '最大绝对误差':f"{values['max_absolute_error']:,.2f}",
            })
        st.markdown('#### 金额组成项一致性')
        st.dataframe(pd.DataFrame(component_rows),hide_index=True,width='stretch')
        st.dataframe(table('balance_anomaly_samples.csv'),hide_index=True)
        st.caption('总申购已包含收益；总赎回已包含消费与转出，不能重复加减。非连续用户观测单独统计，异常记录保留。')
        st.subheader('异常日期（全历史 IQR 描述性标记）')
        st.dataframe(subset.loc[subset.total_purchase_outlier|subset.total_redeem_outlier])
    with tabs[3]:
        dist=json.loads((OUTPUT_DIR/'user_distributions.json').read_text(encoding='utf8'))
        attribute_labels={'sex':'性别编码','city':'城市编码','constellation':'星座'}
        for attr,values in dist.items():
            counts=pd.Series(values,dtype='int64').sort_values(ascending=False)
            distribution=pd.DataFrame({
                '属性取值':counts.index.astype(str),
                '用户数':counts.values,
                '占比':(counts.values/counts.sum()),
            })
            st.markdown(f"#### {attribute_labels.get(attr,attr)}分布")
            chart(px.bar(
                distribution,x='属性取值',y='用户数',text='用户数',
                hover_data={'占比':':.2%'},
                title=f"{attribute_labels.get(attr,attr)}用户分布",
            ))
        st.caption('实际用户画像仅包含性别、城市和星座。未提供年龄、等级或注册日期，不虚构这些字段。首次观测不等于真实注册。')

elif page=='特征分析':
    st.write('目标 lag：1、2、3、7、14、21、28、30 天。滚动窗口：3、7、14、30 天。全部滚动统计先 shift(1)。')
    cutoff=st.selectbox('只用以下训练截止日之前分析',['2014-05-31','2014-06-30','2014-07-31'])
    train=d.loc[:cutoff]; x=make_features(train,True)
    target=st.selectbox('未来目标',['purchase','redeem'])
    correlations=x.corrwith(train['total_'+target]).dropna().sort_values(key=abs,ascending=False).head(20)
    chart(px.bar(x=correlations.values,y=correlations.index,orientation='h',title='训练期：滞后特征与当天目标的相关性'))
    cols=['total_purchase','total_redeem','net_flow','daily_active_users','avg_balance','yield_rate','shibor_overnight']
    chart(px.imshow(train[cols].corr(),zmin=-1,zmax=1,color_continuous_scale='RdBu_r',title='训练期相关性热力图（描述性，不自动选特征）'))
    acf=pd.DataFrame({'lag':range(1,61),'purchase':[train.total_purchase.autocorr(i) for i in range(1,61)],'redeem':[train.total_redeem.autocorr(i) for i in range(1,61)]})
    chart(px.bar(acf,x='lag',y=['purchase','redeem'],barmode='group',title='训练期 ACF：观察周周期与持续性'))
    st.subheader('树模型重要性 / Ridge 标准化系数绝对值')
    importance=table('feature_importance.csv')
    if len(importance):
        model=st.selectbox('解释模型',importance.Model.unique()); target2=st.selectbox('解释目标',TARGETS)
        part=importance[(importance.Model==model)&(importance.target==target2)].nlargest(20,'importance')
        chart(px.bar(part,x='importance',y='feature',orientation='h'))
        st.caption('这是模型实际输出的全历史拟合重要性。相关性不代表因果，重要性也不代表验证期贡献；带 diagnostic 的模型仅用于解释。')
        if len(part):st.write('该模型最依赖的输入：'+ '、'.join(part.feature.head(5))+ '。滞后/滚动特征反映历史水平与周期；日历反映事前已知的时间结构。')
    st.info('扩展特征包含滞后的用户行为、性别比例和利率。整段未来预测中这些未知外生量保持起点值，效果通过同样设定的回测检验。')

elif page=='模型训练':
    st.write('交互实验只写入当前会话，不覆盖已冻结的正式实验与九月提交结果。')
    c1,c2,c3=st.columns(3)
    with c1:
        target=st.selectbox('Target',['Both','Purchase','Redeem'])
        kind=st.selectbox('Model',['Baseline','Ridge','RandomForest','GradientBoosting','HistGradientBoosting','XGBoost','LightGBM','TimeSeries'])
        if kind=='Baseline':kind=st.selectbox('Baseline 类型',BASELINES)
    with c2:
        n=st.slider('n_estimators',50,300,120,10);depth=st.slider('max_depth',2,10,3)
    with c3:
        lr=st.slider('learning_rate',.01,.15,.04,.01);leaf=st.slider('min_samples_leaf',2,30,10)
    enriched=st.checkbox('加入滞后行为与利率特征',False)
    months=st.multiselect('按月固定起点回测', ['2014-06','2014-07','2014-08'],default=['2014-06','2014-07'])
    if '2014-08' in months:
        st.warning('对八月反复试参会使八月失去独立检验资格。正式输出不使用这里的实验选型。')
    if st.button('开始训练与验证',type='primary'):
        if not months:
            st.error('至少选择一个月份')
        else:
            results=[]; details=[]; resolved=[]
            spec={'kind':kind,'n_estimators':n,'max_depth':depth,'learning_rate':lr,'min_samples_leaf':leaf,'enriched':enriched}
            targets=TARGETS if target=='Both' else [target.lower()]
            with st.spinner('按时间训练并递归预测整月…'):
                for start,end in FOLDS:
                    if start[:7] not in months:continue
                    history=d.loc[d.index<pd.Timestamp(start)]; actual=d.loc[start:end]
                    models={t:fit_model(history,t,spec) for t in targets}
                    resolved.extend(t+': '+m.resolved_name for t,m in models.items())
                    pred=recursive_forecast(history,actual.index,models)
                    results.append({'month':start[:7],**evaluate(actual,pred)})
                    details.append(error_frame(actual,pred))
            st.session_state['experiment']=(pd.DataFrame(results),pd.concat(details),resolved,target)
    if 'experiment' in st.session_state:
        results,details,resolved,target=st.session_state['experiment']
        st.success('训练和递归验证已完成');st.write('实际模型',sorted(set(resolved)))
        if target!='Both':st.info('仅训练所选目标；另一目标以 Weekly 递归生成，作为跨目标滞后输入，其指标不代表所选模型。')
        st.dataframe(results,hide_index=True)
        chart(px.line(results,x='month',y=['Purchase_MAPE','Redeem_MAPE'],markers=True,title='不同 Fold 验证误差（非训练 loss）'))
        chart(px.line(details,x=details.index,y=['actual_purchase','pred_purchase','actual_redeem','pred_redeem']))
        score_note()

    st.divider()
    st.subheader('正式模型')
    st.write('正式模型依次执行五个月滚动回测、日历 Ridge、月度总量与日形状、事件距离全参数搜索，最后生成两个目标隔离提交。')
    st.caption('参数空间固定：2014-04 至 2014-07 用于选型，2014-08 仅作独立门禁。完整训练耗时较长，不会覆盖原融合方案的 prediction_201409.csv。')
    if st.button('运行正式模型并生成两个隔离 CSV',type='primary',key='run_event_distance'):
        from src.c3_full_pipeline import run_full_pipeline
        try:
            with st.status('正在运行正式模型完整链路…',expanded=True) as event_status:
                event_result=run_full_pipeline(d,callback=event_status.write)
                event_status.update(label='正式模型完整链路已完成',state='complete',expanded=False)
            st.session_state['event_distance_manifest']=event_result['manifest']
            st.success('正式模型训练与目标隔离完成，两个 CSV 均已生成。')
        except (FileNotFoundError,ValueError) as exc:
            st.error(str(exc))
    if EVENT_DISTANCE_PURCHASE_ONLY_PATH.exists() and EVENT_DISTANCE_REDEEM_ONLY_PATH.exists():
        manifest=st.session_state.get('event_distance_manifest')
        if manifest is None and EVENT_DISTANCE_MANIFEST_PATH.exists():
            manifest=json.loads(EVENT_DISTANCE_MANIFEST_PATH.read_text(encoding='utf8'))
        if manifest and 'event_distance_selected' in manifest:
            selected=pd.DataFrame.from_dict(manifest['event_distance_selected'],orient='index')
            selected.index.name='target'
            show_cols=[c for c in ['ridge_weight','hw_weight','event_scale','tune_delta','winning_months','worst_month_delta','holdout_delta'] if c in selected]
            st.dataframe(selected[show_cols].reset_index(),hide_index=True)
            with st.expander('查看正式模型训练阶段'):
                st.dataframe(pd.DataFrame({'阶段':[
                    '滚动回测','日历模型选型','月度总量与日形状选型',
                    '正式模型参数选型','目标隔离导出',
                ]}),hide_index=True,width='stretch')
        if C3_EVENT_DISTANCE_VALIDATION_PATH.exists():
            validation=read_csv(str(C3_EVENT_DISTANCE_VALIDATION_PATH),C3_EVENT_DISTANCE_VALIDATION_PATH.stat().st_mtime_ns)
            validation['date']=pd.to_datetime(validation['date'])
            available_folds=validation['fold'].drop_duplicates().tolist()
            selected_fold=st.selectbox(
                '实际值与预测值对比月份',available_folds,
                index=len(available_folds)-1,key='event_validation_fold',
            )
            validation_part=validation[validation['fold']==selected_fold].set_index('date')
            fold_role='独立检验' if selected_fold=='2014-08' else '开发期回放'
            st.caption(f'{selected_fold} · {fold_role}；九月没有真实标签，因此不绘制虚构的九月实际值。')
            event_chart_data=validation_part.reset_index().melt(
                id_vars='date',value_vars=['actual_purchase','pred_purchase'],
                var_name='series',value_name='amount',
            )
            event_chart_data['series']=event_chart_data['series'].map({'actual_purchase':'实际申购','pred_purchase':'预测申购'})
            chart(px.line(event_chart_data,x='date',y='amount',color='series',markers=True,title='申购：实际值 vs 正式模型预测值'))
            event_chart_data=validation_part.reset_index().melt(
                id_vars='date',value_vars=['actual_redeem','pred_redeem'],
                var_name='series',value_name='amount',
            )
            event_chart_data['series']=event_chart_data['series'].map({'actual_redeem':'实际赎回','pred_redeem':'预测赎回'})
            chart(px.line(event_chart_data,x='date',y='amount',color='series',markers=True,title='赎回：实际值 vs 正式模型预测值'))
        event_tabs=st.tabs(['仅替换申购','仅替换赎回'])
        for event_tab,event_path in zip(event_tabs,[EVENT_DISTANCE_PURCHASE_ONLY_PATH,EVENT_DISTANCE_REDEEM_ONLY_PATH]):
            with event_tab:
                event_frame=read_submission(str(event_path),event_path.stat().st_mtime_ns)
                st.dataframe(event_frame,hide_index=True)
                st.download_button(
                    f'下载 {event_path.name}',event_path.read_bytes(),event_path.name,'text/csv',
                    key=f'download_{event_path.stem}',
                )
        st.caption(f'输出目录：{OUTPUT_DIR}')

elif page=='模型评估':
    tab=st.radio('评价数据',['八月独立检验','六月七月开发期'],horizontal=True)
    frame=table('holdout_comparison.csv' if tab=='八月独立检验' else 'model_comparison.csv')
    st.dataframe(frame.sort_values(['Weighted_Error','Simulated_Score'],ascending=[True,False]),hide_index=True)
    chart(px.bar(frame,x='Model',y=['Purchase_MAPE','Redeem_MAPE'],barmode='group'))
    chart(px.scatter(frame,x='Weighted_Error',y='Simulated_Score_30day_equivalent',text='Model'))
    score_note()
    selection=json.loads((OUTPUT_DIR/'selection.json').read_text(encoding='utf8'))
    st.subheader('开发期冻结权重');st.json(selection['weights'])
    st.caption('开发期拟合融合的分数有选型乐观偏差；泛化判断看八月冻结融合。融合不保证在每个后续月份超过最佳单模型。')

elif page=='滚动回测':
    frame=table('metrics.csv')
    names=st.multiselect('模型',frame.Model.unique(),default=['Weekly','RandomForest','Ensemble_prequential'])
    part=frame[frame.Model.isin(names)]
    chart(px.line(part,x='Period',y='Weighted_Error',color='Model',markers=True))
    chart(px.line(part,x='Period',y='Redeem_MAPE',color='Model',markers=True))
    st.dataframe(part,hide_index=True);score_note()
    st.markdown('每月起点重新训练，整月递归预测；七月融合只用六月的误差，八月融合只用六月和七月的误差。六月融合没有既往折，预先使用 Weekly。')
    audit=json.loads((OUTPUT_DIR/'leakage_audit.json').read_text(encoding='utf8'))
    st.subheader('时序与数据泄漏审计')
    audit_rows=[]
    for fold,item in enumerate(audit,1):
        audit_rows.append({
            'Fold':fold,
            '训练截止日':item['train_end'][:10],
            '状态':'通过' if item['status']=='passed' else item['status'],
            '检查项':'；'.join(item['checks']),
        })
    st.dataframe(pd.DataFrame(audit_rows),hide_index=True,width='stretch')
    limitations=list(dict.fromkeys(item.get('limitations','') for item in audit if item.get('limitations')))
    for limitation in limitations:
        st.info(f'审计边界：{limitation}')

elif page=='未来预测':
    forecast_name='prediction_201409.csv'
    pred=table(forecast_name);pred['date']=pd.to_datetime(pred.date.astype(str),format='%Y%m%d')
    st.info('2014年9月1日—30日 · 无真实标签 · 输出四舍五入为整数；预测方式取决于所选方案')
    chart(px.line(pred,x='date',y=['purchase','redeem'],markers=True,title='未来30天申购与赎回预测'))
    st.dataframe(pred,hide_index=True)
    a,b=st.columns(2)
    a.download_button('Download CSV（含表头）',(OUTPUT_DIR/forecast_name).read_bytes(),forecast_name.split('/')[-1],'text/csv')
    b.download_button('下载无表头提交版',(OUTPUT_DIR/forecast_name.replace('.csv','_no_header.csv')).read_bytes(),forecast_name.split('/')[-1].replace('.csv','_no_header.csv'),'text/csv')
    validation_path=OUTPUT_DIR/'prediction_validation.json'
    validation=json.loads(validation_path.read_text(encoding='utf8'))
    st.subheader('预测产物验证与模型来源')
    cards=st.columns(4)
    cards[0].metric('输出行数',validation.get('rows','—'))
    cards[1].metric('有限数值','通过' if validation.get('finite') else '未通过')
    cards[2].metric('非负检查','通过' if validation.get('nonnegative') else '未通过')
    cards[3].metric('启用模型数',len(validation.get('negative_raw_predictions',{})))
    if validation.get('weights'):
        st.markdown('#### 冻结融合权重')
        weights=weight_table(validation['weights'])
        st.dataframe(weights.style.format({'权重':'{:.2%}'}),hide_index=True,width='stretch')
    negative_rows=flatten_counts(validation.get('negative_raw_predictions',{}))
    if negative_rows:
        st.markdown('#### 原始负预测截断统计')
        st.dataframe(pd.DataFrame(negative_rows),hide_index=True,width='stretch')
    st.caption(f"来源：{validation.get('source','—')}")

elif page=='误差分析':
    data=table('validation_predictions.csv');data['date']=pd.to_datetime(data.date)
    a,b=st.columns(2)
    model=a.selectbox('分析模型',data.Model.unique(),index=list(data.Model.unique()).index('Ensemble_prequential'))
    fold=b.selectbox('月份 / Fold',[1,2,3],index=2)
    e=data[(data.Model==model)&(data.Fold==fold)].set_index('date')
    st.caption('蓝色/黄色/红色分级分别标识相对误差超过10%、20%、30%。')
    def color_error(value):
        """相对误差阈值着色。"""
        return 'background-color: '+('#fecaca' if value>.3 else '#fde68a' if value>.2 else '#dbeafe' if value>.1 else 'transparent')
    st.dataframe(e.style.map(color_error,subset=['purchase_relative_error','redeem_relative_error']))
    for title,cols in [('实际值与预测值',['actual_purchase','pred_purchase','actual_redeem','pred_redeem']),('每日绝对误差',['purchase_abs_error','redeem_abs_error']),('每日相对误差',['purchase_relative_error','redeem_relative_error']),('每日模拟得分',['daily_simulated_score'])]:
        chart(px.line(e,x=e.index,y=cols,title=title))
    chart(px.scatter(e,x='purchase_relative_error',y='redeem_relative_error',hover_name=e.index.strftime('%Y-%m-%d'),title='Purchase vs Redeem Error'))
    chart(px.bar(e.groupby(e.index.dayofweek)[['purchase_relative_error','redeem_relative_error']].mean(),barmode='group',title='按星期平均误差'))
    groups=np.where(e.index.day<=5,'月初1–5日',np.where(e.index.day>=26,'月末26日起','月中'))
    chart(px.bar(e.groupby(groups)[['purchase_relative_error','redeem_relative_error']].mean(),barmode='group',title='按月初/月末平均误差'))
    st.subheader('高误差日期（任一目标超过30%）');st.dataframe(e.loc[(e.purchase_relative_error>.3)|(e.redeem_relative_error>.3)])
    for line in error_insights(e):st.write('• '+line)
    st.download_button('导出本组误差明细',e.to_csv().encode('utf-8-sig'),'error_analysis.csv','text/csv')
    score_note()
