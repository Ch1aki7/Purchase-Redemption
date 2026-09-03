# Purchase-Redemption 资金流入流出预测系统

本项目基于天池“资金流入流出预测”赛题，实现余额宝每日申购额和赎回额预测。项目已整理为适合提交到 GitHub 的版本：**代码、报告、说明文档可以提交；原始数据文件不提交**。

## 一、项目说明

目标：根据 2013-07-01 至 2014-08-31 的历史数据，预测 2014-09-01 至 2014-09-30 每日资金申购总额 `purchase` 和赎回总额 `redeem`。

本项目包含：

- 数据读取与预处理
- 用户余额一致性校验，并输出数据质量报告
- 每日申购/赎回聚合
- 多维特征工程：周期傅里叶编码、法定节假日/调休工作日、月初月末工作日、滞后/滚动统计、历史同期及用户行为日级聚合（全部基于历史值，无数据泄露）
- 浅层 XGBoost 三种子集成建模，滚动预测评估（验证集与9月预测逻辑一致）
- 2014 年 8 月验证集评估
- 2014 年 9 月滚动预测
- Streamlit 可视化展示
- 课程需求分析文档

## 二、目录结构

```text
Purchase-Redemption/
├── Purchase Redemption Data/                 # 原始赛题数据目录（不提交 Git）
│   ├── user_balance_table.csv               # 用户申购赎回表（核心大表，约 150MB）
│   ├── user_profile_table.csv               # 用户信息表（性别/城市/星座）
│   ├── mfd_day_share_interest.csv            # 余额宝收益率表（万份收益/7日年化）
│   ├── mfd_bank_shibor.csv                   # Shibor 同业拆借利率表（8个期限）
│   ├── comp_predict_table.csv               # 赛题官方提交示例文件
│   └── README.txt                            # 数据说明文档
│
├── src/                                      # 核心源码包
│   ├── __init__.py                           # 包初始化文件，使 src 可作为模块导入
│   ├── config.py                             # 路径配置（统一管理所有输入输出路径）
│   ├── data_loader.py                        # 数据读取（加载4张原始表并做基本清洗）
│   ├── preprocess.py                         # 预处理与特征工程（生成214列日级宽表）
│   ├── calendar_features.py                  # 节假日、调休与工作日位置特征
│   ├── behavior_features.py                  # 用户交易结构的历史聚合特征
│   ├── ensemble.py                           # 可稳定序列化的多种子模型封装
│   ├── train.py                              # 模型训练与验证（多种子 XGBoost 集成）
│   ├── backtest.py                           # 扩展窗口递归滚动回测与实验策略
│   ├── predict.py                            # 9月滚动预测与提交文件生成
│   └── evaluate.py                           # 评估函数（MAPE与模拟评分计算）
│
├── output/                                   # 运行结果输出目录（运行后生成）
│   ├── daily_features.csv                    # 预处理后的日级训练数据（含全部特征）
│   ├── data_quality_report.csv               # 数据质量报告（余额一致性校验）
│   ├── validation_prediction.csv             # 8月验证集预测结果（含真实值/预测值/误差）
│   ├── tc_comp_predict_table.csv             # 9月最终预测提交文件（无表头，赛题格式）
│   └── tc_comp_predict_table_with_header.csv # 9月预测带表头版本（便于查看）
│
├── models/                                   # 训练好的模型文件目录（运行后生成）
│   ├── model_purchase.pkl                    # 申购预测模型（多种子 XGBoost 集成）
│   ├── model_redeem.pkl                      # 赎回预测模型（多种子 XGBoost 集成）
│   └── feature_columns.json                  # 特征列名清单（预测时保持特征一致）
│
├── app.py                                    # Streamlit 可视化系统主入口
├── run_all.py                                # 一键运行完整流程（预处理→训练→预测）
├── requirements.txt                          # Python 依赖清单
├── Purchase Redemption Data.zip              # 原始数据压缩包（备份，不提交 Git）
├── 设计报告.md                                # 面向课程指导书要求的项目设计报告
├── 项目二_需求分析与最小方案.md                # 课程需求分析与方案设计文档
├── LICENSE                                   # 开源许可证（MIT）
└── .gitignore                                # Git 忽略规则
```

### 关键文件说明

**源码模块（`src/`）**

| 文件 | 作用 | 关键函数 |
|---|---|---|
| `config.py` | 集中管理所有文件路径，避免硬编码 | `DAILY_FEATURES_PATH`、`MODEL_PURCHASE_PATH` 等 |
| `data_loader.py` | 读取4张原始 CSV，解析日期，返回 DataFrame | `load_user_balance()`、`load_share_interest()` 等 |
| `preprocess.py` | 聚合每日总量，生成214列日级宽表并保存 | `preprocess_and_feature_engineering()` |
| `calendar_features.py` | 生成法定节假日、调休和月内工作日位置特征 | `calendar_feature_dict()` |
| `behavior_features.py` | 生成用户交易人数、大额交易和渠道占比的严格历史统计 | `add_behavior_history_features()` |
| `train.py` | 训练3种子浅层 XGBoost 集成模型，用8月做递归验证 | `build_xgboost()` |
| `backtest.py` | 比较模型、目标形式和实验策略 | `run_backtest()` |
| `predict.py` | 对9月30天做滚动预测，生成赛题提交格式文件 | `rolling_predict()` |
| `evaluate.py` | 计算 MAPE 和模拟评分，供训练和可视化共用 | `evaluate_prediction()` |

**运行结果（`output/`）**

| 文件 | 作用 | 生成阶段 |
|---|---|---|
| `daily_features.csv` | 预处理后的日级宽表，每行一天，共214列；正式模型使用其中125个可预测特征 | `preprocess.py` 运行后 |
| `data_quality_report.csv` | 数据质量报告，统计余额一致性校验结果 | `preprocess.py` 运行后 |
| `validation_prediction.csv` | 8月验证集预测对比，含 date、purchase_true/pred、redeem_true/pred | `train.py` 运行后 |
| `tc_comp_predict_table.csv` | 9月预测提交文件，无表头，格式为 `report_date,purchase,redeem`，共30行 | `predict.py` 运行后 |
| `tc_comp_predict_table_with_header.csv` | 带表头版本，便于人工查看和可视化展示 | `predict.py` 运行后 |

**模型文件（`models/`）**

| 文件 | 作用 |
|---|---|
| `model_purchase.pkl` | 申购预测模型，`joblib` 序列化的多种子 XGBoost 集成对象 |
| `model_redeem.pkl` | 赎回预测模型，同上 |
| `feature_columns.json` | 训练时的特征列名顺序，预测时必须保持一致 |

**原始数据（`Purchase Redemption Data/`）**

| 文件 | 说明 |
|---|---|
| `user_balance_table.csv` | 约 280 万条用户交易记录，覆盖 2013.07.01~2014.08.31，是聚合每日总量的数据源 |
| `user_profile_table.csv` | 约 2.8 万用户的人口属性信息，本项目的每日总量预测未直接使用 |
| `mfd_day_share_interest.csv` | 余额宝每日万份收益和7日年化收益率，作为外部特征 |
| `mfd_bank_shibor.csv` | 8 个期限的 Shibor 利率，作为外部特征 |
| `comp_predict_table.csv` | 赛题官方提交示例，参考格式用 |

## 三、代码流程

本章节阐释整个项目的运行逻辑，从原始数据到最终预测结果的完整链路。

### 3.1 整体流程

```
Purchase Redemption Data/          原始数据（4张CSV表）
        ↓
src/preprocess.py                  预处理 + 特征工程
        ↓
output/daily_features.csv          日级宽表（214列，正式模型使用125个特征）
        ↓
src/train.py                       训练集成模型
        ↓
models/model_purchase.pkl          申购模型
models/model_redeem.pkl            赎回模型
models/feature_columns.json        特征列名契约
        ↓
output/validation_prediction.csv   8月验证集评估
        ↓
src/predict.py                     9月滚动预测
        ↓
output/tc_comp_predict_table.csv   赛题提交文件（30天预测）
```

### 3.2 各阶段详解

**阶段1：预处理与特征工程（`src/preprocess.py`）**

- 从 `user_balance_table.csv` 按 `report_date` 聚合，得到每日 `purchase` 和 `redeem` 总量
- 校验 `tBalance = yBalance + total_purchase_amt - total_redeem_amt`，输出 `output/data_quality_report.csv`
- 合并收益率表和 Shibor 利率表
- 生成 214 列日级宽表；正式模型筛选其中 125 个预测时可获得且回测有效的特征
- 节假日特征同时区分法定假日、调休工作日、节前/节后和月内第几个工作日
- 用户行为与4/8/12周同星期统计保留在宽表供分析，但因五个月滚动回测降分，默认模型不直接使用
- 输出 `output/daily_features.csv`

**阶段2：模型训练（`src/train.py`）**

- 读取 `daily_features.csv`，按日期切分：训练集 2013.08~2014.07，验证集 2014.08
- 对 `purchase` 和 `redeem` 分别训练 `EnsembleModel`（3个不同随机种子的浅层 XGBoost 平均）
- 训练时目标做 `log1p` 变换（日总量上亿分，取对数稳定方差）
- XGBoost 使用深度2、学习率0.02、最多2000棵树，并用验证集做100轮早停
- **验证集评估采用滚动预测**：逐天预测8月1日→8月31日，预测值回填为下一天 lag 特征，与 9 月预测逻辑完全一致，避免使用8月真实历史导致分数虚高
- 用 `joblib.dump` 保存模型到 `models/`，同时保存特征列名到 `feature_columns.json`

**阶段3：9月滚动预测（`src/predict.py`）**

- 用 `joblib.load` 加载模型，读取 `feature_columns.json` 对齐特征顺序
- 逐天滚动预测：第 1 天用 8 月最后一天的特征预测，预测值回填为第 2 天的 lag 特征
- 9 月的收益率和 Shibor（未来未知）用 8 月均值填充
- 平滑修正：模型预测 × 0.99 + 近 7 日均值 × 0.01，缓解滚动漂移
- 周期融合：申购采用 `100% XGBoost`；赎回采用 `50% XGBoost + 50% 最近12个同星期日均值`。周期锚点只修正输出，不进入模型递归 lag
- 默认不做固定月末倍率修正；申购×1.2、赎回×1.3 只保留在滚动回测中作为实验策略
- 八月验证完成后，使用选出的树数量在截至 2014-08-31 的全部数据上重拟合正式模型
- 输出 `output/tc_comp_predict_table.csv`（无表头，赛题格式）和带表头版本

### 3.3 models 目录文件说明

`models/` 下的三个文件由 `src/train.py` 生成，供 `src/predict.py` 使用：

| 文件 | 生成方式 | 用途 |
|---|---|---|
| `model_purchase.pkl` | `joblib.dump(EnsembleModel)` 保存申购预测模型 | `predict.py` 加载后做 9 月申购预测 |
| `model_redeem.pkl` | 同上，保存赎回预测模型 | `predict.py` 加载后做 9 月赎回预测 |
| `feature_columns.json` | `json.dump(feature_cols)` 保存 125 个入模特征及顺序 | `predict.py` 按此顺序构造特征矩阵，保证与训练一致 |

**为什么不直接用，要保存？** 训练耗时几分钟，预测只需几秒。保存后可复用模型，不用每次预测都重训。

**`feature_columns.json` 的作用**：训练和预测必须用完全一致的特征顺序，否则模型报错或预测错乱。重新预处理后特征列可能变化，此时需重训模型，这个 JSON 是训练与预测之间的「契约」。

### 3.4 数据放置要求

原始赛题数据不提交到 Git，运行前请把天池数据解压到：

```text
Purchase Redemption Data/
```

必需文件：

```text
user_balance_table.csv      # 核心大表，约 280 万条，缺它无法训练
user_profile_table.csv
mfd_day_share_interest.csv
mfd_bank_shibor.csv
```

如果缺少 `user_balance_table.csv`，模型无法训练。

**用户画像说明**：项目已读取 `user_profile_table`，但最终预测目标是每日申购/赎回总量，且测试期未来用户画像难以稳定映射到每日资金流；因此当前主模型未直接使用用户画像特征，而是在数据资产说明和可视化展示中保留其作用边界。

**余额一致性说明**：指导书要求关注 `tBalance = yBalance + total_purchase_amt - total_redeem_amt`。项目已在 `src/preprocess.py` 中实现该校验，并输出 `output/data_quality_report.csv`。该报告只用于数据质量说明，不删除样本、不改变训练和预测逻辑。

## 四、安装依赖

建议使用 Python 3.8 或以上版本。

```bash
pip install -r requirements.txt
```

正式流程需要 `xgboost`；`lightgbm`、`scikit-learn` 模型保留用于回测对照和前端实验。

## 五、一键运行

在项目根目录执行：

```bash
python run_all.py
```

运行完成后会生成：

```text
output/daily_features.csv
output/data_quality_report.csv
output/validation_prediction.csv
output/tc_comp_predict_table.csv
output/tc_comp_predict_table_with_header.csv
models/model_purchase.pkl
models/model_redeem.pkl
models/feature_columns.json
```

文件说明：

| 文件 | 作用 |
|---|---|
| `daily_features.csv` | 预处理和特征工程后的日级训练数据 |
| `data_quality_report.csv` | 数据质量报告，记录余额一致性校验统计 |
| `validation_prediction.csv` | 2014 年 8 月验证集预测结果，用于评估模型效果 |
| `tc_comp_predict_table.csv` | 2014 年 9 月最终预测提交文件，默认无表头 |
| `tc_comp_predict_table_with_header.csv` | 带表头版本，便于查看和展示 |

## 六、启动可视化系统

```bash
streamlit run app.py
```

页面包括：

1. 数据探索：展示历史申购/赎回趋势、用户分布、收益率变化、Shibor 利率变化和数据质量报告
2. 模型训练与评估：允许选择 XGBoost、LightGBM、RandomForest、GradientBoosting 及参数进行真实训练，并展示损失曲线、交叉验证和验证集对比
3. 预测结果展示：展示 2014 年 9 月预测结果，并下载无表头提交文件，同时展示验证集每日误差和得分
4. 误差分析：展示验证集绝对误差、相对误差、每日得分、零分日、高估/低估方向和改进提示

侧边栏提供“一键运行完整流程”按钮，会执行 `run_all.py` 并重新生成 `output/` 与 `models/`；网页中的交互训练结果只保存在当前 Streamlit 会话中，不覆盖正式结果文件。

如果首次运行 Streamlit 出现邮箱提示，直接按回车跳过即可。

## 七、如何判断预测是否可靠

2014 年 9 月真实值属于隐藏测试集，本地无法直接知道最终预测是否完全正确。因此本项目使用 2014 年 8 月作为验证集，**采用与 9 月一致的滚动预测逻辑**（逐天预测，预测值回填为下一天的 lag 特征），评估模型在未来 30 天上的真实表现。

**关于评分公式**：指导书明确「误差与得分之间的计算公式不公布」，只保证 error=0 得 10 分、error>0.3 得 0 分、单调递减。本项目采用线性近似公式 `score = max(0, 10×(1-error/0.3))` 做课程项目内的模拟评估，**该分数为内部模拟值，非天池官方成绩**。

当前验证集评估结果（线性近似公式，满分 10）：

| 指标 | 数值 |
|---|---|
| 申购 MAPE | 15.79% |
| 赎回 MAPE | 17.67% |
| 申购评分 | 5.00 |
| 赎回评分 | 4.74 |
| **模拟总分** | **4.8558** |
| 申购零分日（误差>30%） | 4 天 |
| 赎回零分日（误差>30%） | 6 天 |

**关于分数**：该分数是在**无数据泄露、验证集滚动预测**条件下得到的本地近似结果。早期版本曾因验证集使用 8 月真实历史而虚高至 8.14，已修正。当前浅层 XGBoost 方案的 8 月严格递归验证分为 **4.8558**；更重要的是，三种子五个月整体分数为 **4.7038**，高于上一版周期融合的 **4.5721（约 +2.9%）**。固定月末倍率的五个月整体仅 **4.4879**，仍只作为实验策略。

评估结果主要查看：

- 申购 MAPE
- 赎回 MAPE
- 模拟总分
- 真实值与预测值折线图是否接近

最终 `tc_comp_predict_table.csv` 可用于赛题平台或课程结果展示。

## 八、提交到 GitHub

本项目已经配置 `.gitignore`，默认不会提交：

- 原始数据文件
- 运行生成的模型文件
- 运行生成的输出结果
- 压缩包
- Python 缓存文件

日常更新代码只需：

```bash
git add .
git commit -m "你的提交说明"
git push
```

若首次复现本仓库到新的本地目录，用 `git clone https://github.com/Ch1aki7/Purchase-Redemption.git` 即可。

## 九、原项目文件说明

当前推荐运行主流程为：

```bash
python run_all.py
streamlit run app.py
```

项目已集成端到端流程：数据预处理 → 特征工程 → 多种子浅层 XGBoost 集成训练 → 滚动预测 → 可视化展示。默认预测不做固定月末倍率修正；历史单月实验分数不作为主流程选型依据，应以多月份滚动回测结果评估泛化能力。

## 十、滚动回测

为避免只在 2014 年 8 月上调参导致结论偏乐观，可以对多个历史月份执行扩展窗口回测。每个回测折只使用目标月份之前的数据；目标月内逐日递归预测，并把预测值回填为后续日期的滞后特征。

快速回测默认使用 1 个浅层 XGBoost 随机种子，评估 2014 年 4 月至 8 月：

```bash
python -m src.backtest
```

如需使用与正式模型相同的 3 种子集成：

```bash
python -m src.backtest --n-seeds 3
```

也可以指定月份：

```bash
python -m src.backtest --months 2014-06 2014-07 2014-08
```

回测会同时比较纯模型、默认周期融合和月末上调实验策略，输出：

```text
output/rolling_backtest_predictions.csv
output/rolling_backtest_metrics.csv
```

早停月份取自目标月之前；确定迭代轮数后，模型会使用目标月之前的全部历史重新拟合，避免用待评估月份选择树数量。

### 正式回测结果（三种子）

| 月份 | 纯模型 | 周期融合（默认） | 月末倍率实验 |
|---|---:|---:|---:|
| 2014-04 | 4.2028 | **4.5743** | 3.6449 |
| 2014-05 | 5.1299 | 5.1833 | **5.2816** |
| 2014-06 | 3.8982 | 3.8274 | **3.9101** |
| 2014-07 | 4.9804 | **4.9940** | 4.5841 |
| 2014-08 | 4.5807 | 4.9075 | **4.9732** |
| **五个月整体** | **4.5650** | **4.7038** | **4.4879** |

默认策略只对赎回进行 50% 周期融合，五个月整体比纯模型提高约 3.0%；申购保持纯模型，避免周期均值拉低波动月份。周期融合并非每个月都占优，但整体最好；固定月末倍率整体低于纯模型，因此只作为实验策略保留。

### 本轮按序优化结论

| 步骤 | 五个月滚动回测结论 | 默认采用 |
|---|---|---|
| 细化节假日、调休和工作日位置 | 相比旧日历特征有效 | 是 |
| 新增4/8/12周同星期均值、中位数、标准差 | 共线和过拟合，整体降分 | 否，仅保留供分析 |
| 用户人数、大额交易、渠道占比历史聚合 | 当前样本量下整体降分 | 否，仅保留供分析 |
| 预测相对星期基线的残差 | 4月明显失稳，整体约3.15 | 否，保留 `--target-mode residual` 实验入口 |
| Ridge / LightGBM / XGBoost 对比 | Ridge约2.98；默认LightGBM约4.65；浅层XGBoost更稳 | 选择浅层XGBoost |
| 目标分别融合 | 申购不融合，赎回50%模型+50%周期锚点整体最好 | 是 |

以上数值均为项目自定义线性近似评分，不是天池官方成绩。
