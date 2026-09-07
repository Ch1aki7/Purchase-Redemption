"""从真实实验文件生成样本证据、对照结论与复现报告。"""
import json
import numpy as np
import pandas as pd
from .config import OUTPUT_DIR,TARGETS
from .metrics import evaluate
from .multistep_samples import SAMPLE_DIR

def markdown_table(frame):
    """不依赖可选tabulate，生成可读Markdown。"""
    result=['|'+'|'.join(map(str,frame.columns))+'|','|'+'|'.join(['---']*len(frame.columns))+'|']
    for row in frame.itertuples(index=False,name=None):
        result.append('|'+'|'.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/') for v in row)+'|')
    return '\n'.join(result)

def build_report():
    """所有收益、样本数和有效组从真实结果计算，不硬编码结论。"""
    frozen=json.loads((SAMPLE_DIR/'frozen_selection.json').read_text(encoding='utf8'))
    metrics=pd.read_csv(SAMPLE_DIR/'development_metrics.csv')
    ablation=pd.read_csv(SAMPLE_DIR/'ablation_comparison.csv')
    groups=pd.read_csv(SAMPLE_DIR/'effective_sample_groups.csv')
    errors=pd.read_csv(SAMPLE_DIR/'development_predictions.csv',parse_dates=['date'])
    ledger=pd.read_csv(SAMPLE_DIR/'samples/sample_ledger.csv',parse_dates=['origin','label_date'])
    x=pd.read_csv(SAMPLE_DIR/'samples/features.csv.gz',index_col='sample_id')
    y=pd.read_csv(SAMPLE_DIR/'samples/normalized_labels.csv.gz',index_col='sample_id')
    scales=pd.read_csv(SAMPLE_DIR/'samples/origin_scales.csv.gz',index_col='sample_id')
    selected=pd.DataFrame()
    for t,name in frozen['chosen'].items():
        part=errors[errors.Experiment==name].set_index('date')
        selected['actual_'+t]=part['actual_'+t]
        selected['pred_'+t]=part['pred_'+t]
    selected.to_csv(SAMPLE_DIR/'selected_development_predictions.csv',index_label='date')
    actual=selected[['actual_purchase','actual_redeem']].rename(columns={'actual_purchase':'total_purchase','actual_redeem':'total_redeem'})
    pred=selected[['pred_purchase','pred_redeem']].rename(columns={'pred_purchase':'purchase','pred_redeem':'redeem'})
    selected_metrics=evaluate(actual,pred)
    pd.DataFrame([selected_metrics]).to_csv(SAMPLE_DIR/'selected_development_metrics.csv',index=False)
    horizon=[]
    for t in TARGETS:
        err=(selected['actual_'+t]-selected['pred_'+t]).abs()/selected['actual_'+t].abs().clip(lower=1)
        for name,mask in [('1-7',selected.index.day<=7),('8-14',(selected.index.day>=8)&(selected.index.day<=14)),('15-31',selected.index.day>=15)]:
            horizon.append({'target':t,'horizon_group':name,'actual_days':int(mask.sum()),'MAPE':float(err[mask].mean()*100)})
    pd.DataFrame(horizon).to_csv(SAMPLE_DIR/'selected_horizon_metrics.csv',index=False)
    feature_use=[];selected_usage=[]
    for path in (SAMPLE_DIR/'fit_receipts').rglob('*.json'):
        for receipt in json.loads(path.read_text(encoding='utf8')):
            for prefix in ['component_','behavior_','cohort_']:
                used={k:v for k,v in receipt['used_split_features'].items() if k.startswith(prefix)}
                feature_use.append({'Period':path.parent.name,'Experiment':receipt['experiment'],'target':receipt['target'],'head':receipt['head'],'group':prefix,'input_columns':sum(c.startswith(prefix) for c in receipt['feature_columns']),'actually_split_features':len(used),'total_splits':sum(used.values())})
            if frozen['chosen'][receipt['target']]==receipt['experiment']:
                selected_usage.append({'Period':path.parent.name,'target':receipt['target'],'head':receipt['head'],'experiment':receipt['experiment'],'rows_to_fit':receipt['rows_actually_passed_to_fit'],'real_label_dates':receipt['unique_label_dates'],'origins':receipt['unique_origins'],'input_features':receipt['feature_count'],'actually_split_features':receipt['used_feature_count']})
    pd.DataFrame(feature_use).to_csv(SAMPLE_DIR/'feature_usage_by_group.csv',index=False)
    pd.DataFrame(selected_usage).to_csv(SAMPLE_DIR/'selected_training_usage.csv',index=False)
    examples=[]
    removals=pd.read_csv(SAMPLE_DIR/'sample_group_removal.csv')
    for _,group in groups[groups.evidence_of_usefulness].iterrows():
        target=group.target;name=frozen['chosen'][target]
        eligible=removals[(removals.target==target)&(removals.removed_group==group.removed_group)&(removals.delta_after_removal_pp>0)]
        if eligible.empty:continue
        period=eligible.sort_values('Period').iloc[-1].Period
        receipt_path=SAMPLE_DIR/'fit_receipts'/period/(name+'_'+target+'_members.csv.gz')
        members=pd.read_csv(receipt_path,parse_dates=['origin','label_date']).drop_duplicates('sample_id')
        cutoff=pd.Timestamp(period+'-01')-pd.Timedelta(days=1)
        age=(cutoff-members.origin).dt.days
        masks={'recent_0_30':age<=30,'middle_31_90':(age>30)&(age<=90),'older_91_plus':age>90,'month_end_origins':members.origin.dt.is_month_end}
        pool=members[masks[group.removed_group]].sort_values(['origin','horizon'])
        subset=pool[pool.horizon.isin([1,7,14,30])]
        if subset.empty:subset=pool
        for _,sample in subset.tail(4).iterrows():
            sid=sample.sample_id
            examples.append({'target':target,'experiment':name,'validation_month':period,'group':group.removed_group,'sample_id':sid,'origin':sample.origin,'horizon':sample.horizon,'label_date':sample.label_date,'actual_target_amount':float(y.loc[sid,target]*scales.loc[sid,target]),'target_weekday_feature':float(x.loc[sid,target+'_target_weekday_28']),'component_direct_share':float(x.loc[sid,'component_direct_share']),'high_balance_purchase_share':float(x.loc[sid,'cohort_high_balance_purchase_share']),'evidence_scope':'属于开发期剔除实验有收益的组，不证明此单行具有因果贡献'})
    pd.DataFrame(examples,columns=['target','experiment','validation_month','group','sample_id','origin','horizon','label_date','actual_target_amount','target_weekday_feature','component_direct_share','high_balance_purchase_share','evidence_scope']).to_csv(SAMPLE_DIR/'effective_sample_examples.csv',index=False)
    overall=metrics.groupby('Experiment')[['Purchase_MAPE','Redeem_MAPE','Weighted_Error']].mean().sort_values('Weighted_Error').reset_index()
    # 额外与原随机森林的同月份历史实验对照，避免只和较弱的新基准比较。
    reference_path=OUTPUT_DIR/'optimization/development_metrics.csv'
    reference=''
    if reference_path.exists():
        old=pd.read_csv(reference_path)
        reference_rows=old[old.Model=='RF_original'][['Period','Purchase_MAPE','Redeem_MAPE','Weighted_Error']].copy()
        reference_rows['Experiment']='Original_recursive_RF'
        selected_rows=[]
        for period in metrics.Period.unique():
            row={'Period':period,'Experiment':'Selected_traced_multistep'}
            for t,name in frozen['chosen'].items():
                col=t.capitalize()+'_MAPE';row[col]=float(metrics[(metrics.Period==period)&(metrics.Experiment==name)].iloc[0][col])
            row['Weighted_Error']=(.45*row['Purchase_MAPE']+.55*row['Redeem_MAPE'])/100
            selected_rows.append(row)
        comparison=pd.concat([reference_rows,pd.DataFrame(selected_rows)],ignore_index=True)
        comparison.to_csv(SAMPLE_DIR/'original_reference_comparison.csv',index=False)
        reference=markdown_table(comparison[['Period','Experiment','Purchase_MAPE','Redeem_MAPE','Weighted_Error']])
    lines=['# 真实多步训练样本：入模证据与有效性实验','',
    '## 本轮实际做了什么','',
    f"生成{len(ledger):,}条起点—预测距离样本，来自{ledger.origin.nunique()}个历史起点和{ledger.label_date.nunique()}个真实标签日期。13组方案分别在五月、六月、七月训练和验证，并为两个目标保存实际fit证据。",'',
    '数据来自项目真实CSV。总申购=直接申购+收益入账；总赎回=消费+转入余额+转入银行卡。用户群根据每个起点之前56天的最后观测余额划分高余额、普通、近期未观测/尚未出现三组；阈值和标签组归属不读取未来行为。最后观测余额不等于当天真实余额。','',
    '先构建再训练还不够：每次fit记录sample_id、输入矩阵摘要、标签摘要、实际列数、每行权重、模型迭代数以及树实际分裂使用的特征。样本成员文件可以逐行核验。','',
    '## 时间边界与有效样本量','',
    '每条样本满足 feature_observed_max_date <= origin < label_date <= 当前折训练截止日。未来仅使用公历。训练/推理特征一致性及未来明细扰动不影响特征均有测试。扩展样本行不是独立真实天数；同标签日的多起点样本按重复次数调整训练权重。','',
    '本轮选择仍只使用2014-07-31及之前的标签。八月之前已被观察，因此后续八月称冻结后事后审计，不声称完全未接触。','',
    '## 对照过程','',
    'A核心特征+每周起点+平方损失；B仅改绝对误差损失；C再增加交易分项/行为/群体特征；D将标签改为交易分项后加总；E改为用户群后加总；F改密集起点；G仅对相同密集样本中的月末起点加权；H加入相对误差权重；I/J分别在H基础上改分项/分群；K按预测距离分组；L/M改为120/60天近期窗口。','',
    '首次试跑发现G同时改变了起点范围，已中止、修正并完整重跑。有效结果对应output/multistep_corrected_develop.log，且测试保证F与G的样本ID完全一致。','',
    '## 各方案开发期月均指标','',markdown_table(overall),'',
    '这里的指标用于选型，包含开发期乐观偏差。相对误差为小数，MAPE为百分比。','',
    '## 单因素对照结果','',markdown_table(ablation[['experiment','parent','target','mean_mape_delta_pp','improved_months','consistent_benefit']]),'',
    'delta是子实验减父实验MAPE的百分点。平均下降且至少2/3个月改善才标为consistent_benefit。C的扩展特征是一个组合，其收益不能单独归因于某一个字段；树分裂次数也不是因果贡献。','',
    '## 冻结选择及其实际入模证据','',json.dumps(frozen['chosen'],ensure_ascii=False),'',
    markdown_table(pd.DataFrame(selected_usage)),'',
    '逐次证据：fit_receipts/月份/实验_目标.json 和对应_members.csv.gz。actual split features为真实树节点统计；传入模型但未被分裂使用的列也如实保留。','',
    '## 与原随机森林的同月对照','',reference,'',
    '## 删除哪类样本会变差','',markdown_table(groups),'',
    'mean_error_increase_pp为删组后MAPE变化，正数表示删除后更差。至少两个验证月更差且平均变差才记录为有作用的证据。无可删除样本或样本不足的情况标为not_testable，不算作有效性证据。','',
    'effective_sample_examples.csv列出这些组中确实参与拟合的真实样本ID、原始目标金额和部分特征。它们只是有作用组的成员示例，不能据此断言每一条训练记录单独有效。选型和剔除都在同一开发期，结论属于开发期消融证据，不是因果证明。','',
    '## 不同预测距离表现','',markdown_table(pd.DataFrame(horizon)),'',
    '## 冻结后八月审计']
    if (SAMPLE_DIR/'august_comparison.csv').exists():
        comparison=pd.read_csv(SAMPLE_DIR/'august_comparison.csv')
        lines += ['',markdown_table(comparison[['Model','Purchase_MAPE','Redeem_MAPE','Weighted_Error','Simulated_Score']]),'','模拟分不是官方真实评分。',json.dumps(json.loads((SAMPLE_DIR/'release_check.json').read_text(encoding='utf8')),ensure_ascii=False)]
    else:
        lines += ['','尚未进行八月审计；以上结论只来自七月底之前的开发数据。']
    lines += ['','## 复现与文件','',
    '~~~bash','python -m src.sample_experiment --stage develop','python -m src.sample_experiment --stage audit','python -m src.sample_report','python -m unittest discover -s tests -v','~~~','',
    '前端“多步样本验证”页可以查账本、标签原始金额、实际拟合成员、真实分裂特征、对照结果和有效样本组。','',
    'samples/仅含开发期数据；final_training_samples/供最终九月重训，允许使用八月，两者严格分目录。默认九月CSV不会被本实验覆盖。']
    (SAMPLE_DIR/'sample_experiment_report.md').write_text('\n'.join(lines),encoding='utf8')
    return overall,groups

if __name__=='__main__':
    a,b=build_report();print(a.to_string(index=False));print(b.to_string(index=False))
