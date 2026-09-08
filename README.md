# 资金流入流出预测系统

## 项目背景

依据《大数据技术项目实训》指导书项目二，使用余额宝用户交易、用户画像、收益率及SHIBOR预测平台每日资金流，支持流动性、现金储备与资金配置分析。历史2013-07-01至2014-08-31，预测2014-09-01至09-30。

已实际运行数据审计、特征工程、13种候选方案三折回测、独立目标训练、融合和30天预测，提供八页Streamlit系统。九月没有真实标签，因此没有九月真实误差或官方得分。

## 数据说明

程序自动读取 data/ 下的CSV（排除processed）。首次运行会将 Purchase Redemption Data/ 中五个CSV复制到 data/raw，源文件保持不变。

|表|实际规模|关联与用途|
|---|---:|---|
|user_balance_table|2,840,421 × 18|按用户日记录聚合平台资金流|
|user_profile_table|28,041 × 4|user_id多对一关联性别、城市、星座|
|mfd_day_share_interest|427 × 3|按日期关联每日收益、七日收益|
|mfd_bank_shibor|294 × 9|按日期关联隔夜到一年各期限报价|
|comp_predict_table|3 × 3，无表头|格式样例，隔离，绝不参与训练|

真实映射：report_date/mfd_date → date；total_purchase_amt → purchase；total_redeem_amt → redeem；tBalance/yBalance → balance/previous_balance；mfd_daily_yield → yield_rate；Interest_O_N → shibor_overnight。完整columns、dtypes、head、统计、缺失率、日期范围与映射见 output/data_quality_report.json。

金额保留原始单位，指导书未明确单位换算，不擅自换算为元。category1–4 各缺失约93.88%，不进入模型。没有年龄、等级或注册日期，不虚构字段。余额仅1条相差100，原样保留。总申购已含收益，总赎回已含消费与转出，不能重复加减。详见 [质量报告](output/data_quality_report.md)。

## 项目结构

~~~text
Purchase-Redemption/
├── app.py                         # 八页Web
├── run_all.py                     # 一键流水线
├── start_web.bat                  # Windows启动
├── requirements.txt
├── requirements-tested.txt       # 实际验证版本
├── README.md
├── data/
│   ├── raw/                      # 五个CSV的原始副本
│   └── processed/
│       ├── daily_balance.csv
│       ├── features.csv
│       └── validation_report.csv
├── models/
│   ├── purchase/*.joblib
│   ├── redeem/*.joblib
│   └── manifest.json             # 参数、权重、训练截止
├── output/
│   ├── prediction_201409.csv
│   ├── prediction_201409_no_header.csv
│   ├── validation_predictions.csv
│   ├── metrics.csv
│   ├── model_comparison.csv
│   ├── holdout_comparison.csv
│   ├── feature_importance.csv
│   ├── error_analysis.csv
│   ├── error_insights.txt
│   ├── data_quality_report.json / .md
│   ├── balance_consistency.json
│   ├── balance_anomaly_samples.csv
│   ├── leakage_audit.json
│   ├── prediction_validation.json
│   ├── experiment_report.md
│   ├── pipeline.log
│   └── *.html                    # 离线Plotly图
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── behavior_features.py
│   ├── calendar_features.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── models.py
│   ├── metrics.py
│   ├── backtest.py
│   ├── ensemble.py
│   ├── train.py
│   ├── predict.py
│   ├── c3_full_pipeline.py       # 正式模型编排与当前数据适配
│   ├── event_distance.py         # 输出校验与目标隔离
│   ├── formal_model_core/        # 已迁入当前分支的正式模型核心
│   └── utils.py
└── tests/
    ├── test_pipeline.py
    ├── test_artifacts.py
    └── check_web.py
~~~

## 安装

Python 3.10+。实际验证为Windows、Python 3.13；其他版本根据requirements.txt解析兼容依赖，未逐版本测试。

~~~bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

精确复现建议Python 3.13安装requirements-tested.txt。依赖版本改变后请重新训练，避免旧joblib模型兼容问题。预处理读取约158MB交易表，建议至少4GB可用内存，8GB更稳妥。

可选：python -m pip install xgboost lightgbm。缺少这两个包时交互训练明确回退HistGradientBoosting；缺少statsmodels时TimeSeries明确回退Weekly。默认流程不依赖XGBoost、LightGBM或SHAP。

## 一键运行

~~~bash
python run_all.py
~~~

顺序：预处理 → 特征校验 → 回测 → 全历史重训 → 最终预测。生成结果可覆盖，原始数据不会修改。如更换数据或候选模型，请完整重跑，避免混用旧selection.json。

## 数据预处理

~~~bash
python -m src.preprocess
python -m src.feature_engineering
python -m src.eda
~~~

缺失关键金额、无法解析的日期、日期缺口或重复主键会明确报错。余额异常仅报告，不删除。全样本IQR仅描述性标记，不参与清洗或建模。日志与异常样本保存在output。

## 模型训练

~~~bash
python -m src.train
~~~

没有selection.json时先回测；已有时复用六月七月冻结权重。两目标独立训练、保存。候选参数在 src/models.py 的 model_specs() 统一声明。

## 历史回测

~~~bash
python -m src.backtest
~~~

|Fold|训练截止|验证区间|用途|
|---|---|---|---|
|1|2014-05-31|2014-06-01—06-30|开发|
|2|2014-06-30|2014-07-01—07-31|开发|
|3|2014-07-31|2014-08-01—08-31|独立检验|

每月固定起点重训，整月递归预测；不是月内每日获得真实值后做单步预测。融合按目标搜索最佳单模型/两模型凸组合（0.1步长）。在线融合六月预设Weekly，七月只用六月，八月只用六月七月。最终权重用六月七月，八月不反向调参。开发期拟合融合成绩有选型乐观偏差。

## 生成最终预测

~~~bash
python -m src.predict
~~~

output/prediction_201409.csv 含date,purchase,redeem表头；日期YYYYMMDD，30天，金额四舍五入为int64。另有prediction_201409_no_header.csv，与原始样例无表头格式一致。逐模型记录负预测截断数，拒绝NaN与inf。

第1天lag来自历史；以后来自真实历史加已预测值，两目标同步递归。整段预测不接收验证真实金额。

## 启动系统

~~~bash
streamlit run app.py
# 或
python -m streamlit run app.py
~~~

访问 http://localhost:8501 ，或双击start_web.bat。页面：项目概览、数据探索、特征分析、模型训练、模型评估、滚动回测、未来预测、误差分析。

图表支持hover、缩放、图例开关，EDA支持时间范围。训练支持目标、模型、核心超参数、扩展特征和验证月选择。交互实验保留在会话中，不覆盖正式模型与提交。单目标实验另一目标以Weekly递归生成跨目标lag，页面明确标注。没有适用loss曲线时展示真实Fold误差，不伪造loss。

## 模型说明

- Yesterday、Weekly、MA7/14/30、SameWeekday：六类递归基线。
- Ridge：填充与标准化只fit训练集。
- RandomForest、GradientBoosting、HistGradientBoosting：独立目标回归、固定随机种子。
- RandomForest_enriched：附加滞后画像、行为与利率，检验其预测价值。
- Redeem_smooth：更大叶节点和较低学习率的树候选，重点检查赎回稳定性。
- TimeSeries：Holt-Winters加性周季节、阻尼趋势。
- Ensemble：不同目标独立选择组合和权重。

每个候选均训练两个独立模型。回归目标使用固定1e8缩放改善数值尺度，输出还原；该常量不由全样本估计。HistGradientBoosting禁用内部随机验证early stopping。解释使用真实树重要性或Ridge标准化系数绝对值，SHAP不是强制依赖。

## 特征说明

日历包括年/月/日、星期、月内日期、周数、周末、月初/月末/季初/季末、距月初/月末天数和星期/月内日期/月的周期编码。

lag为1/2/3/7/14/21/28/30天。rolling均值、最小、最大、中位数为3/7/14/30天，样本标准差为7/14/30天。必须先shift(1)，最初30天暖启动不参加回归拟合。

每日聚合含金额、净流入、人数与比例、均值/中位余额、首次观测比例、性别比例。模型只用有限预定义滞后列；城市、星座分布仅EDA。总申购含收益，所以“活跃”包含仅收益入账者；首次观测不等于注册。

利率仅forward fill，再lag_1。未来行为、收益率、SHIBOR固定预测起点最后已知值，回测使用同一策略。未来日历可事先知道；没有可靠节假日/调休清单，不虚构假日。

## 评价指标

relative_error = abs(pred-actual) / max(abs(actual),epsilon)，epsilon=1个原始金额单位。输出MAE、RMSE、MAPE（百分比）、相对误差和近零目标数量。分母下限防止除零；近零值仍需结合MAE判断。

Weighted_Error = 0.45 × Purchase平均相对误差 + 0.55 × Redeem平均相对误差。

模拟评分 = 10 × max(0,1-relative_error/0.3)，按天求和后45%/55%加权。**模拟评分不代表官方真实评分公式。** 官方仅给单调性和端点，不公开完整映射。31天原始总分与30天等效分分别列出。

## 防止数据泄漏

1. 不打乱时间；验证严格晚于训练，每月重训。
2. validate_no_leakage()检查唯一递增连续日期、无重叠、历史不越过八月底；传入特征时逐值核对合法lag/rolling。
3. 目标、行为和利率均滞后；imputer/scaler仅fit训练集。
4. 特征集合预先声明，相关性只在所选训练截止日前分析。
5. 参数和融合只用六月七月；八月仅评估；九月样例隔离。
6. 递归函数只接收历史及未来日期，结束后才计算真实误差。
7. 上日报价与起点固定外生量是保守假设，仍无法证明原始报价没有事后修订。
8. 静态画像缺少生效时间，假定历史可得；检查器不能证明未知的数据源属性。

## 实验结果

八月独立检验：

|模型|Purchase MAPE|Redeem MAPE|Weighted Error|
|---|---:|---:|---:|
|冻结融合|14.59%|16.96%|15.89%|
|随机森林|14.55%|16.27%|15.50%|
|扩展随机森林|14.66%|16.45%|15.65%|
|Weekly|25.12%|24.41%|24.73%|

冻结融合申购MAE约39,016,630，赎回MAE约45,094,130，金额单位与原始数据一致。精确MAE/RMSE见holdout_comparison.csv。31天加权模拟分158.24，30天等效模拟分153.14。

开发期融合加权误差19.61%，最佳单模型22.45%；八月融合略逊于单独随机森林，不能声称融合稳定更好。保留开发期方案，避免拿八月重新调权重。详见 [实验分析](output/experiment_report.md) 与 model_comparison.csv、metrics.csv。

## 最终预测结果

- output/prediction_201409.csv：恰好30天。
- 申购：50% SameWeekday + 50% RandomForest_enriched。
- 赎回：20% SameWeekday + 80% RandomForest_enriched。
- 模型用截至八月底历史重训，权重来自六月七月。
- 最终基础模型原始负预测均为0，无NaN/inf。
- 九月无标签，不报告九月真实MAPE、官方积分或未经验证的置信区间。

## 测试

~~~bash
python -m unittest discover -s tests -v
python tests/check_web.py
~~~

测试覆盖聚合、余额异常、lag、rolling、训练/推理特征一致、未来扰动不改变过去特征、重叠/九月/日期缺口拒绝、30天预测、非负和负值计数、零值指标、独立目标融合。另检验真实聚合与原始数据一致，以及八页Web与实际训练按钮。

## 项目答辩支持

1. 为什么是时间序列？每天目标存在趋势、周期和相邻日依赖。
2. 为什么不能随机划分？未来阶段会进入训练，破坏真实预测起点。
3. 为什么lag？表达过去水平、周周期和跨目标关联。
4. 为什么rolling先shift？否则包含正在预测的当天真值。
5. 为什么重点Redeem？题目赋予55%权重，独立优化并按加权误差比较。
6. 为什么Walk-Forward？检查不同月份与资金规模变化下的稳定性。
7. 为什么当前模型？依据六月七月选型，有记录，八月独立检查不足。
8. 如何避免泄漏？输入边界、移位、训练期拟合、选择期隔离及对抗测试。
9. 未来lag来自哪里？历史加已预测值，逐日递归，两目标同步。
10. 有何不足？历史短、增长趋势、递归误差、未知外生量和画像时间戳缺失。
11. 更多数据如何改进？多年份、真实节假日、直接多步、分群预测、嵌套验证、预测区间与业务损失。

## 项目不足与改进方向

427天不足以可靠学习年度季节性。总申购含收益限制行为解释；未来利率固定可能错过市场变化；静态画像缺生效时间。后续可引入更多历史、核实的节假日及报价发布时间，用新的未参与选型的留出期评价改进。不能反复优化八月后仍称其独立检验。

## 阶段验收

Phase 1目录/PDF；2质量审计；3余额与聚合；4真实EDA；5因果特征；6基线；7回归与时序模型；8三折回测；9对比；10赎回平滑参数及独立权重；11融合；12最终预测；13八页Web；14测试、说明和脚本。实际结果和运行日志在output。
