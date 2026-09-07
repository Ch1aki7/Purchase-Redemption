from pathlib import Path
import importlib.metadata as metadata
import json
import pandas as pd
root=Path.cwd();out=root/'output'
packages=['pandas','numpy','scikit-learn','matplotlib','plotly','streamlit','joblib','statsmodels']
(root/'requirements-tested.txt').write_text('# 实测环境 Python 3.13，用于复现实验\n'+'\n'.join(f'{p}=={metadata.version(p)}' for p in packages)+'\n',encoding='utf8')
audit=json.loads((out/'data_quality_report.json').read_text(encoding='utf8'))
(out/'initial_data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
report=['# 数据质量与字段口径报告','','## 数据清单与关联关系','','|表|行数|列数|日期范围|','|---|---:|---:|---|']
for key,v in audit.items():
 report.append(f"|{key}|{v['shape'][0]:,}|{v['shape'][1]}|{' 至 '.join(v.get('date_range',[])) or v.get('role','静态画像')}|")
report+=['','交易表通过 user_id 多对一关联静态用户表；按 date 求和后，再与收益率和 SHIBOR 以日期一对一关联。画像没有年龄、等级、注册日期，不制造这些字段。','','## 数据检查','','- 四张主表无完全重复行，交易 user_id/date 与画像 user_id 主键均无重复。','- 交易用户均有画像；目标金额和余额无负值、缺失或非有限值。','- category1–4 各缺失2,666,682条，约93.88%；不进入预测。','- 427个交易日连续；SHIBOR缺133天，其中周末116天，其余17天未核实为法定节假日。','- comp_predict_table.csv 是无表头3行格式样例，不是真值。','','## 余额口径实证','','每条记录验证 tBalance = yBalance + total_purchase_amt - total_redeem_amt。容差为1个原始金额单位，2,840,421条记录中仅1条误差100，异常比例约0.0000352%。另对2,795,780条用户连续日记录，以上一天实际余额验证，同样仅1条异常。44,641条首次或非连续观测不能用前一条记录充当昨天，单独统计。','','全部记录满足以下关系：','','- total_purchase_amt = direct_purchase_amt + share_amt','- direct_purchase_amt = purchase_bal_amt + purchase_bank_amt','- total_redeem_amt = consume_amt + transfer_amt','- transfer_amt = tftobal_amt + tftocard_amt','','因此收益已在总申购中，消费和转出已在总赎回中，重复加减会制造错误。剩余100单位误差原因无法确定；可能是结算修正、记账时点差异、未列明调整或原始记录问题，这些不是已证实原因。异常原样保留，详见 balance_anomaly_samples.csv。','','## 信息边界','','原始文件仅复制到 data/raw。金额保留原始单位，PDF未明确单位换算。利率只按历史日期 forward fill，不做 backward fill；输入模型再滞后一天，整段预测固定起点最后已知报价。最早无历史值保留缺失，imputer仅在训练期拟合。','','活跃定义为总申购或总赎回大于0，包含只有收益入账的用户，不能解释为主动购买。首次观测用户比例不等于注册用户比例。']
(out/'data_quality_report.md').write_text('\n'.join(report),encoding='utf8')
dev=pd.read_csv(out/'model_comparison.csv');held=pd.read_csv(out/'holdout_comparison.csv')
weights=json.loads((out/'selection.json').read_text(encoding='utf8'))['weights']
s=held[held.Model=='Ensemble_prequential'].iloc[0]
cols=['Model','Purchase_MAPE','Redeem_MAPE','Weighted_Error','Simulated_Score_30day_equivalent']
def md_table(df):
 lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
 for _,r in df.iterrows():
  lines.append('|'+str(r.Model)+'|'+'|'.join(f'{r[c]:.4f}' for c in cols[1:])+'|')
 return '\n'.join(lines)
text='''# 实验结果与答辩分析

## 实验协议

历史从2013-07-01开始。三折为2014年6月、7月、8月，各折完整预测下个月30或31天，月内不读取真实目标更新lag。每个候选模型分别拟合申购、赎回，候选参数在回测前固定。

六月和七月用于模型与特征方案选择、两模型凸组合权重搜索（0.1步长）；八月独立检验。在线融合七月只使用六月已结束折，八月使用六月七月，六月无前序折时预设Weekly。

## 开发期比较（61天，存在选择乐观偏差）

'''+md_table(dev)+'''

## 八月独立检验（31天）

'''+md_table(held)+f'''

冻结融合申购 MAE {s.Purchase_MAE:,.2f}，RMSE {s.Purchase_RMSE:,.2f}；赎回 MAE {s.Redeem_MAE:,.2f}，RMSE {s.Redeem_RMSE:,.2f}。金额单位与原始数据一致。

八月申购模拟总分 {s.Purchase_Score:.4f}，赎回模拟总分 {s.Redeem_Score:.4f}，加权模拟总分 {s.Simulated_Score:.4f}（31天）；30天等效模拟分 {s.Simulated_Score_30day_equivalent:.4f}。模拟评分不代表官方真实评分。九月没有标签，不报告九月误差或得分。

## 选择与解释

冻结权重：{json.dumps(weights,ensure_ascii=False)}。

'''+'''申购和赎回权重不同，反映两者的历史规律和偏差不同。开发期融合加权误差19.61%，低于最佳单模型随机森林22.45%；但八月融合15.89%，略逊于单独随机森林15.50%，不能声称融合稳定优于所有单模型。

随机森林在开发期赎回方面优于多数基线；可能因为其结合了滞后、近期水平和日历效应，而长期同星期均值对增长中的赎回水平反应慢。这是与结果一致的解释，不是因果证明。增加画像和利率后未稳定超过核心特征版本，更多特征并不必然带来收益。

赎回优化包括额外平滑树参数候选、独立赎回权重搜索和0.55加权比较，没有使用八月成绩反向调参。

## 八月冻结融合误差诊断

'''+(out/'error_insights.txt').read_text(encoding='utf8')+'''

## 局限与改进

427天且平台存在增长趋势，年周期信息有限；递归误差会传播。缺少可靠假日/调休清单或未来利率，不虚构这些特征。静态画像没有属性生效时间，只能假定历史期可用。无法证明原始报价没有事后修订。

更多历史可支持直接多步模型、增长阶段处理、分群层次预测、外生情景、预测区间及现金缺口损失。下一轮优化应增加新的未参与选择的留出月份，不能反复用八月调参后仍称其独立检验。
'''
(out/'experiment_report.md').write_text(text,encoding='utf8')
