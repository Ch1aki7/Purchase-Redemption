# 余额宝申购赎回预测 — PySpark数据预处理脚本说明书

---

## 目录

- [1. 项目背景](#1-项目背景)
- [2. 数据说明](#2-数据说明)
- [3. 脚本总览](#3-脚本总览)
- [4. 01_data_exploration.py — 数据探索分析](#4-01_data_explorationpy--数据探索分析)
- [5. 02_feature_engineering.py — 特征工程](#5-02_feature_engineeringpy--特征工程)
- [6. 03_train_data_prepare.py — 训练数据准备](#6-03_train_data_preparepy--训练数据准备)
- [7. 运行方式](#7-运行方式)
- [8. 特征清单](#8-特征清单)
- [9. 注意事项](#9-注意事项)

---

## 1. 项目背景

本竞赛要求预测余额宝用户在 **2014年9月1日—30日（共30天）** 每天的 **申购总额（purchase）** 和 **赎回总额（redeem）**，金额精确到分。

### 评估指标

采用积分式计算，分两步：

1. 计算每天申购和赎回的 **相对误差**：
   $$Purchase_i = \frac{|predicted_i - actual_i|}{actual_i}, \quad Redeem_i = \frac{|predicted_i - actual_i|}{actual_i}$$

2. 通过一个**单调递减函数**将误差映射为每日得分（误差=0 → 10分，误差>0.3 → 0分），最终得分加权求和：

   $$\text{总积分} = \text{申购预测得分} \times 45\% + \text{赎回预测得分} \times 55\%$$

---

## 2. 数据说明

> 数据目录：`Purchase Redemption Data/`

| 文件名 | 行数 | 说明 |
|--------|------|------|
| `user_profile_table.csv` | ~28,000 | 用户基本信息：性别、城市、星座 |
| `user_balance_table.csv` | ~2,840,000 | 用户每日申购赎回流水（2013-07-01 ~ 2014-08-31） |
| `mfd_day_share_interest.csv` | 427 | 余额宝万份收益和七日年化收益率 |
| `mfd_bank_shibor.csv` | 294 | 上海银行间同业拆放利率（8个期限） |
| `comp_predict_table.csv` | 3 | 提交样例格式 |

### 2.1 user_balance_table 关键字段

| 字段 | 含义 | 单位 |
|------|------|------|
| `total_purchase_amt` | 今日总购买量 = 直接购买 + 收益 | 分 |
| `total_redeem_amt` | 今日总赎回量 = 消费 + 转出 | 分 |
| `direct_purchase_amt` | 今日直接购买量 | 分 |
| `purchase_bal_amt` | 支付宝余额购买量 | 分 |
| `purchase_bank_amt` | 银行卡购买量 | 分 |
| `consume_amt` | 消费总量 | 分 |
| `transfer_amt` | 转出总量 | 分 |
| `tftobal_amt` | 转出到支付宝余额 | 分 |
| `tftocard_amt` | 转出到银行卡 | 分 |
| `share_amt` | 今日收益 | 分 |
| `category1~4` | 四个消费类目 | 分 |
| `tBalance` / `yBalance` | 今日余额 / 昨日余额 | 分 |

> **数据约束**：`今日余额 = 昨日余额 + 今日申购 - 今日赎回`，不会出现负值。
> `consume_amt=0` 时四个类目字段为空。

### 2.2 收益率表字段

| 字段 | 含义 |
|------|------|
| `mfd_daily_yield` | 万份收益，即1万块钱的收益（元） |
| `mfd_7daily_yield` | 七日年化收益率（%） |

### 2.3 Shibor 表字段

| 字段 | 期限 |
|------|------|
| `Interest_O_N` | 隔夜 |
| `Interest_1_W` | 1周 |
| `Interest_2_W` | 2周 |
| `Interest_1_M` | 1个月 |
| `Interest_3_M` | 3个月 |
| `Interest_6_M` | 6个月 |
| `Interest_9_M` | 9个月 |
| `Interest_1_Y` | 1年 |

---

## 3. 脚本总览

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  01_data_exploration.py                         │
│  ├─ 自动检测 JAVA_HOME                           │
│  ├─ 读取4张原始表                                │
│  ├─ Schema & 基础统计                            │
│  ├─ 缺失值 & 日期对齐检查                         │
│  └─ 输出：exploration_output/daily_aggregation.csv│
│              │                                   │
│              ▼                                   │
│  02_feature_engineering.py                      │
│  ├─ 自动检测 JAVA_HOME                           │
│  ├─ 每日聚合（用户级 → 每日级）                    │
│  ├─ 合并收益率表 + Shibor表                       │
│  ├─ 时间特征 + 滞后特征 + 滚动窗口特征              │
│  ├─ 趋势特征 + 收益率/Shibor衍生特征               │
│  ├─ 用户画像聚合特征                              │
│  └─ 输出：feature_engineering_output/             │
│          engineered_features.csv                  │
│              │                                   │
│              ▼                                   │
│  03_train_data_prepare.py                       │
│  ├─ 自动检测 JAVA_HOME                           │
│  ├─ 从原始数据一站式构建（可独立运行）              │
│  ├─ 划分训练集/验证集                             │
│  ├─ VectorAssembler + StandardScaler             │
│  ├─ 特征相关性分析                                │
│  └─ 输出：modeling_output/                        │
│          train_data.csv, val_data.csv,            │
│          train_scaled.csv, val_scaled.csv         │
│                                                 │
└─────────────────────────────────────────────────┘
```

> 每个脚本启动时会**自动检测并设置 JAVA_HOME**（支持 `C:\Program Files\Microsoft\jdk-17.*` 等常见路径），无需用户手动配置。

---

## 4. 01_data_exploration.py — 数据探索分析

### 功能

对四张原始数据表进行全面的探索性数据分析（EDA），了解数据的基本特征和质量。

### 执行步骤

| 步骤 | 函数 | 说明 |
|------|------|------|
| 1 | `read_tables()` | 读取4张CSV，输出行数和列数 |
| 2 | `explore_schema()` | 打印每张表的Schema |
| 3 | `explore_user_profile()` | 性别/城市Top10/星座分布 |
| 4 | `explore_user_balance()` | 日期范围、每日活跃用户、各金额字段汇总统计（sum/avg/std/max）、零值比例 |
| 5 | `explore_yield_and_shibor()` | 收益率和Shibor的描述性统计和日期范围 |
| 6 | `check_date_alignment()` | 检查三张表日期是否对齐，找出缺失日期 |
| 7 | `check_missing_values()` | 各列NULL值统计，category字段与consume_amt的关联 |
| 8 | `save_exploration_results()` | 保存每日聚合统计CSV |

### 关键输出

```
exploration_output/
└── daily_aggregation.csv    # 427行每日聚合统计：active_users, total_purchase,
                             #   total_redeem, total_share, total_consume 等
```

### 运行验证结果（本地实测）

| 项目 | 数值 |
|------|------|
| 用户总数 | 28,041 |
| 交易记录 | 2,840,421 行 |
| 日期范围 | 2013-07-01 ~ 2014-08-31（427天） |
| 收益率日期 | 427天（全覆盖） |
| Shibor日期 | 294天（周末/节假日缺失，脚本自动前向填充） |
| 性别分布 | 男 51.7% / 女 48.3% |
| consume_amt=0 比例 | 93.88%（category字段NULL原因） |
| 耗时 | ~75秒 |

---

## 5. 02_feature_engineering.py — 特征工程

### 功能

将原始用户级交易数据聚合到每日级别，合并外部宏观数据，构造丰富的特征集。这是最核心的脚本。

### 执行步骤

#### Step 1: 用户级 → 每日级聚合

对每个 `report_date`，汇总所有用户的交易行为：

| 特征类别 | 具体特征 |
|----------|----------|
| 用户行为统计 | `active_users`、`purchase_users`、`redeem_users`、`consume_users`、`transfer_users` |
| 申购相关 | `total_purchase`、`direct_purchase`、`purchase_bal`、`purchase_bank` |
| 赎回相关 | `total_redeem`、`total_consume`、`total_transfer`、`total_tftobal`、`total_tftocard` |
| 收益 & 消费类目 | `total_share`、`cat1_total` ~ `cat4_total` |
| 余额统计 | `total_balance`、`avg_balance`、`std_balance`、`max_balance` |
| 人均指标 | `avg_purchase_per_user`、`avg_redeem_per_user`、`avg_consume_per_user` |
| 比率特征 | `purchase_user_ratio`、`redeem_user_ratio`、`consume_ratio_in_redeem`、`direct_purchase_ratio`、`purchase_bank_ratio`、`net_flow` |

#### Step 2: 合并外部数据

- 左连接 `mfd_day_share_interest`（收益率表）
- 左连接 `mfd_bank_shibor`（Shibor表）
- 对缺失值执行**前向填充**（用最近的有效值填充）

#### Step 3: 时间特征

| 类别 | 特征 |
|------|------|
| 基础时间 | `year`、`month`、`day`、`day_of_week`、`week_of_year`、`quarter` |
| 二值标记 | `is_weekend`、`is_month_start`、`is_month_end`、`is_monday`、`is_friday` |
| 周期编码 | `dow_sin/cos`、`month_sin/cos`、`day_sin/cos`（用三角函数捕获周期性） |

#### Step 4: 滞后特征（Lag Features）

为以下6个核心指标各构造 **1, 2, 3, 7, 14, 21, 28, 30天** 的滞后值 + 1日环比变化：

- `total_purchase`
- `total_redeem`
- `active_users`
- `total_share`
- `total_consume`
- `direct_purchase`

#### Step 5: 滚动窗口特征（Rolling Window）

为5个核心指标各构造 **7, 14, 30天** 窗口的统计量：

| 统计量 | 含义 |
|--------|------|
| `_avg` | 窗口均值 |
| `_std` | 窗口标准差 |
| `_max` | 窗口最大值 |
| `_min` | 窗口最小值 |

#### Step 6: 趋势特征

- **动量比** `momentum_7v30`：7日均值 / 30日均值（短期vs长期趋势）
- **趋势变化** `trend_7v7`：近7日均值 / 前7日均值

#### Step 7: 收益率和Shibor衍生特征

- **收益率**：`yield_change_1d`（日变化）、`yield_7d_change`（七日年化变化）、lag 1/3/7
- **Shibor利差**：`shibor_spread_1M_O_N`、`shibor_spread_1Y_O_N`、`shibor_spread_3M_1M`、`shibor_slope`
- **Shibor变化**：`shibor_on_change`、`shibor_1w_change`

#### Step 8-11: 后处理

- 缺失值填充（滞后/滚动计算导致的前期缺失 → 填0）
- 用户画像按日聚合（男女申购/赎回占比）
- 特征列整理和输出

### 输出

```
feature_engineering_output/
└── engineered_features.csv    # 完整特征工程数据集（~427行 × 150+列）
```

---

## 6. 03_train_data_prepare.py — 训练数据准备

### 功能

从原始数据一站式构建训练数据集，可独立运行（不依赖脚本2的输出）。完成数据集划分、标准化、相关性分析等建模前准备工作。

### 执行步骤

#### Step 1: 一站式构建

整合脚本2的核心逻辑，从原始CSV直接构建每日聚合表并合并外部数据。

#### Step 2: 全量特征构造

构造所有特征（时间/比率/滞后/滚动/趋势/收益率/Shibor），与脚本2逻辑一致。

#### Step 3: 数据集划分

| 数据集 | 日期范围 | 用途 |
|--------|----------|------|
| **训练集** | 2013-07-01 ~ 2014-07-31 | 模型训练 |
| **验证集** | 2014-08-01 ~ 2014-08-31 | 模型调参/验证 |
| **预测集** | 2014-09-01 ~ 2014-09-30 | 最终预测（无真实标签） |

#### Step 4: 描述性统计

输出训练集和验证集的目标变量统计（min/max/mean/std），及按月、按星期几的聚合趋势。

#### Step 5: 建模数据集准备

- **原始特征版**：`VectorAssembler` 将特征列组装为 `features_raw` 向量 → 用于树模型（GBDT/Random Forest/XGBoost）
- **标准化版**：`StandardScaler`（withMean=True, withStd=True） → 用于线性模型（Ridge/Lasso）或神经网络

#### Step 6: 特征相关性分析

计算特征间的皮尔逊相关系数矩阵，识别高度共线的特征对（|r| > 0.9），辅助特征选择。

#### Step 7: 预测模板

生成2014年9月1-30日的预测日期列表。实际预测时需逐日滚动构造特征（第N天的滞后特征依赖前N-1天的预测值）。

### 输出

```
modeling_output/
├── train_data.csv          # 训练集（原始特征，含所有列）
├── val_data.csv            # 验证集（原始特征，含所有列）
├── train_scaled.csv        # 训练集标准化版（仅report_date + 目标 + features_scaled）
└── val_scaled.csv          # 验证集标准化版（仅report_date + 目标 + features_scaled）
```

---

## 7. 运行方式

### 7.1 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.x | 已验证 3.13.3 |
| Java JDK | 17 LTS | PySpark 依赖 JVM，推荐 Microsoft OpenJDK 17 |
| PySpark | 4.x | `pip install pyspark` |
| Pandas | 2.x | `pip install pandas`（用于 CSV 输出） |

### 7.2 一键环境搭建

```powershell
# 1. 安装 Java 17（Windows，需管理员权限）
winget install Microsoft.OpenJDK.17

# 2. 安装 Python 依赖
pip install pyspark pandas

# 3. 验证环境
python -c "from pyspark.sql import SparkSession; print('OK')"
```

> **注意**：步骤1安装完成后需**重新打开终端**使环境变量生效。
> 如果未重启终端，脚本会自动在常见路径下搜索 JDK 并设置 `JAVA_HOME`。

### 7.3 运行脚本

```powershell
# 进入项目目录
cd D:\Daily\BD

# Step 1: 数据探索（可选，了解数据概况，~75秒）
python 01_data_exploration.py

# Step 2: 特征工程（输出完整特征数据集）
python 02_feature_engineering.py

# Step 3: 训练数据准备（可独立运行，不依赖Step 2）
python 03_train_data_prepare.py
```

> 三个脚本都会在启动时打印 `[INFO] 自动设置 JAVA_HOME = ...` 确认 Java 已就绪。

### 7.4 JAVA_HOME 自动检测机制

每个脚本开头内置了以下逻辑，无需用户手动设置环境变量：

```python
_JAVA_PATHS = [
    r"C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot",
    r"C:\Program Files\Java\jdk-17",
    r"C:\Program Files\Java\jdk-11",
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot",
]
if "JAVA_HOME" not in os.environ:
    for p in _JAVA_PATHS:
        if os.path.isdir(p):
            os.environ["JAVA_HOME"] = p
            break
```

如果你的 JDK 安装在其他路径，可以：
- 设置系统环境变量 `JAVA_HOME`（推荐）
- 或修改脚本中的 `_JAVA_PATHS` 列表

### 7.5 CSV 输出说明

由于 Windows 环境下 PySpark 写入本地 CSV 需要 Hadoop 的 `winutils.exe`，脚本改用 **Pandas 写 CSV** 的方式：`DataFrame.toPandas().to_csv()`。

| 优点 | 说明 |
|------|------|
| 无需 Hadoop | 不需要配置 `HADOOP_HOME` 和 `winutils.exe` |
| 单文件输出 | 直接输出一个 csv 文件，而非 Spark 的 `part-*.csv` 目录 |
| UTF-8 BOM | 使用 `utf-8-sig` 编码，Excel 可直接打开不乱码 |

### 7.6 在 Jupyter/交互式环境中

```python
# 在notebook中直接运行
import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot"

from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("test") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# 然后复制脚本中的核心函数到notebook中运行
```

### 7.7 关键Spark配置

脚本中已内置以下优化配置：

```python
.config("spark.sql.shuffle.partitions", "200")
.config("spark.sql.adaptive.enabled", "true")
.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
```

对于2.8M行数据，这些配置通常足够。如果数据量更大，可适当调大 `shuffle.partitions`。

---

## 8. 特征清单

### 最终特征集（约150+列）

| 特征组 | 数量 | 示例 |
|--------|------|------|
| 用户行为 | ~10 | `active_users`, `purchase_user_ratio`, `avg_purchase_per_user` |
| 金额细分 | ~17 | `direct_purchase`, `total_consume`, `net_flow` |
| 余额 | ~5 | `avg_balance`, `std_balance`, `total_balance` |
| 时间 | ~17 | `is_weekend`, `dow_sin`, `month_cos`, `quarter` |
| 收益率 | ~10 | `mfd_daily_yield`, `yield_change_1d`, `yield_lag_7` |
| Shibor | ~14 | `Interest_O_N`, `shibor_spread_1M_O_N`, `shibor_slope` |
| 用户画像 | ~8 | `male_purchase_ratio`, `male_user_ratio` |
| 滞后(lag 1-30d) | ~48 | `total_purchase_lag_1`, `total_redeem_lag_28` |
| 滚动窗口(7/14/30d) | ~60 | `total_purchase_roll_7d_avg`, `total_redeem_roll_30d_std` |
| 趋势 | ~10 | `total_purchase_momentum_7v30`, `total_purchase_trend_7v7` |
| 环比变化 | ~2 | `purchase_change_1d`, `redeem_change_1d` |

### 目标变量

| 变量 | 类型 | 含义 |
|------|------|------|
| `total_purchase` | bigint | 当日总申购额（分） |
| `total_redeem` | bigint | 当日总赎回额（分） |

---

## 9. 注意事项

### 9.1 数据特性

1. **收益计算简化**：采用自然日而非会计日，0点为分界。转入时间与首次显示收益时间的关系遵循题目给定的映射表（周一转入→周三显示收益）。

2. **金额单位**：所有金额以 **分** 为单位（0.01元），预测结果也应为分。

3. **数据脱敏**：数据经过脱敏处理但保持了原始趋势。

### 9.2 建模建议

1. **时间序列特性**：这是典型的时间序列预测问题，特征构造重点在滞后和滚动窗口。建议尝试：
   - 树模型（GBDT/XGBoost/LightGBM）：使用原始特征，对异常值鲁棒
   - 线性模型（Ridge/Lasso）：使用标准化特征，需要处理多重共线性
   - 时序模型（ARIMA/Prophet）：对趋势和季节性建模

2. **滚动预测**：9月的预测需要逐日进行 — 预测第1天（9月1日）后，将其预测值作为第2天的 `lag_1` 特征输入，依次滚动30天。

3. **验证策略**：用8月作为验证集是最合理的选择，因为它紧邻9月且反映了最近的模式。

4. **特征筛选**：脚本3输出了相关性矩阵，建议去除 |r|>0.95 的特征对中的一个以降低共线性。

### 9.3 可能的问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `JAVA_HOME is not set` | PySpark需要Java运行环境 | 安装JDK 17：`winget install Microsoft.OpenJDK.17`，脚本内置了自动检测 |
| `HADOOP_HOME and hadoop.home.dir are unset` | Windows缺少Hadoop的winutils | 脚本已改用Pandas写CSV来规避此问题 |
| `No module named 'pandas'` | 缺少Pandas | `pip install pandas`（脚本用Pandas输出CSV） |
| 收益率/Shibor缺失值 | 非交易日无Shibor数据（周末） | 脚本已做前向填充 |
| 滞后特征前期缺失 | 数据开始的前30天没有足够的滞后历史 | 脚本已填充为0 |
| category字段大量NULL（93.88%） | consume_amt=0时类目为空 | 符合数据说明，非Bug |
| 内存不足 | 2.8M行数据 + 大量窗口计算 | 增大executor内存，或分批处理 |

### 9.4 架构设计说明

**为什么输出用 Pandas 而非 Spark 原生写入？**

在 Windows 上，PySpark 的 `DataFrame.write.csv()` 底层依赖 Hadoop 的 `FileOutputCommitter`，而后者需要 `winutils.exe`（Hadoop 的 Windows 兼容层）。配置 `HADOOP_HOME` + `winutils.exe` 比较繁琐，由于聚合后的数据量很小（427行），使用 `.toPandas().to_csv()` 绕过此限制，效果相同且更简洁。

### 9.5 提交格式

最终提交文件为 `tc_comp_predict_table.csv`：

```csv
report_date,purchase,redeem
20140901,<预测申购额>,<预测赎回额>
20140902,<预测申购额>,<预测赎回额>
...
20140930,<预测申购额>,<预测赎回额>
```

- 30行（9月1日到30日）
- 逗号分隔
- 金额以**分**为单位
