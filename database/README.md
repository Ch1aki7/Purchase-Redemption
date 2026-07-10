# 数据库设计说明

本目录存放资金流入流出预测系统的 **SQLite 数据库 schema 与初始化脚本**。

---

## 目录

- [1. 建表思路](#1-建表思路)
- [2. 分层架构](#2-分层架构)
- [3. 表清单与字段说明](#3-表清单与字段说明)
- [4. 视图](#4-视图)
- [5. 使用方式](#5-使用方式)
- [6. 与项目模块的对应关系](#6-与项目模块的对应关系)

---

## 1. 建表思路

### 1.1 核心原则

| 原则 | 说明 |
|------|------|
| **按数据流向分层** | 原始 → 加工 → 建模 → 预测评估，而非按 CSV 文件名简单堆表 |
| **保留原始数据** | 四张竞赛表原样入库，可追溯、可重新跑 ETL |
| **加工数据与日级聚合** | 427 行日级特征单独存表，避免可视化/建模每次扫 284 万行 |
| **预留系统接口** | 模型注册、预测结果、每日评估等表先建结构，后续模块直接写入 |
| **质量可验证** | 余额公式、日期对齐等检查落库，满足项目「数据一致性」要求 |

### 1.2 为什么选 SQLite

- 团队项目零配置，单文件 `data/purchase_redemption.db` 易共享
- 数据规模（284 万行交易 + 427 天日级）SQLite 完全胜任
- Streamlit / FastAPI 可直接连接
- 分层结构后续可迁移至 PostgreSQL

### 1.3 关键设计决策

| 决策 | 原因 |
|------|------|
| 金额字段用 **INTEGER（分）** | 与竞赛数据一致，避免浮点误差 |
| 日期用 **INTEGER（YYYYMMDD）** | 与原始 CSV 一致，便于 JOIN |
| `user_balance` 建 **日期/用户索引** | 284 万行，按日聚合和按用户查询需要 |
| `daily_features` 由 CSV **动态建表** | 204 列特征，手写 DDL 难维护 |
| Purchase / Redeem **分开建模、合并存储** | 竞赛分开预测，评估时加权合并 |
| 用 **视图** 封装常用查询 | 趋势图、余额校验逻辑固定，减少重复 SQL |

### 1.4 数据流

```
Purchase Redemption Data/*.csv
        │
        ▼  init_db.py 导入
┌───────────────────────────────────────────────────┐
│  原始层: user_profile / user_balance / yield / shibor │
└───────────────────────┬───────────────────────────┘
                        │ PySpark 01/02/03
                        ▼
┌───────────────────────────────────────────────────┐
│  加工层: daily_summary / daily_features           │
│  建模层: ml_train_data / ml_val_data / dataset_split │
└───────────────────────┬───────────────────────────┘
                        │ 模型训练 & 预测（待开发）
                        ▼
┌───────────────────────────────────────────────────┐
│  评估层: model_registry / predictions / daily_evaluation │
└───────────────────────────────────────────────────┘
```

---

## 2. 分层架构

```
database/
├── schema.sql      # DDL：表、索引、视图
├── init_db.py      # 建库 + 导 CSV + 质量检查
└── README.md       # 本文档

data/
└── purchase_redemption.db   # 运行时生成（已 gitignore）
```

| 层级 | 表 | 状态 |
|------|-----|------|
| 原始数据 | `user_profile`, `user_balance`, `mfd_day_share_interest`, `mfd_bank_shibor`, `comp_predict_template` | 已导入 |
| 加工数据 | `daily_summary`, `daily_features`, `daily_features_meta` | 前两表已导入，meta 预留 |
| 建模数据 | `ml_train_data`, `ml_val_data`, `dataset_split`, `model_registry` | 前三表已导入，registry 预留 |
| 预测评估 | `predictions`, `daily_evaluation` | 预留 |
| 数据质量 | `data_quality_report` | 初始化时自动写入 |

---

## 3. 表清单与字段说明

### 3.1 原始数据层

#### `user_profile` — 用户信息

| 字段 | 类型 | 作用 |
|------|------|------|
| `user_id` | INTEGER PK | 用户 ID |
| `sex` | INTEGER | 0=女，1=男 |
| `city` | INTEGER | 城市编码（脱敏） |
| `constellation` | TEXT | 星座 |

#### `user_balance` — 用户申购赎回流水（核心，约 284 万行）

| 字段 | 作用 |
|------|------|
| `user_id`, `report_date` | 用户 + 日期 |
| `t_balance`, `y_balance` | 今日/昨日余额（分） |
| `total_purchase_amt` | 今日总申购 |
| `direct_purchase_amt`, `purchase_bal_amt`, `purchase_bank_amt` | 申购细分 |
| `total_redeem_amt` | 今日总赎回 |
| `consume_amt`, `transfer_amt`, `tftobal_amt`, `tftocard_amt` | 赎回细分 |
| `share_amt` | 今日收益 |
| `category1~4` | 消费类目（consume=0 时为 NULL） |

> 约束：`t_balance = y_balance + total_purchase_amt - total_redeem_amt`（见视图 `v_balance_violations`）

#### `mfd_day_share_interest` — 余额宝收益率

| 字段 | 作用 |
|------|------|
| `mfd_date` | 日期 |
| `mfd_daily_yield` | 万份收益 |
| `mfd_7daily_yield` | 七日年化（%） |

#### `mfd_bank_shibor` — 银行间拆借利率

8 个期限字段：`interest_o_n`（隔夜）~ `interest_1_y`（1 年）。周末/节假日缺失，特征工程需前向填充。

#### `comp_predict_template` — 提交格式模板

2014 年 9 月 1–30 日的 `report_date` / `purchase` / `redeem` 样例。

---

### 3.2 加工数据层

#### `daily_summary` — 每日聚合（427 行）

来自 `01_data_exploration.py` → `exploration_output/daily_aggregation.csv`。

核心字段：`active_users`, `total_purchase`, `total_redeem`, 各细分金额, `avg_balance`。

#### `daily_features` — 完整特征（427 × 204 列）

来自 `02_feature_engineering.py` → `feature_engineering_output/engineered_features.csv`。

含时间、滞后、滚动窗口、收益率、Shibor、用户画像等特征，列由 CSV 表头动态生成。

#### `daily_features_meta` — 特征字典（预留）

记录每个特征的分组与中文说明，供前端特征选择器使用。

---

### 3.3 建模数据层

#### `ml_train_data` / `ml_val_data`

| 表 | 日期范围 | 行数 |
|----|----------|------|
| `ml_train_data` | 2013-07-01 ~ 2014-07-31 | 396 |
| `ml_val_data` | 2014-08-01 ~ 2014-08-31 | 31 |

来自 `03_train_data_prepare.py` → `modeling_output/`。

#### `dataset_split` — 划分标记

| `split_type` | 含义 |
|--------------|------|
| `train` | 训练集 |
| `val` | 验证集（模拟 9 月前最近一个月） |
| `predict` | 2014-09 待预测 30 天 |

#### `model_registry` — 模型注册（预留）

记录模型名称、算法、超参数 JSON、训练/验证积分、模型文件路径等。

---

### 3.4 预测与评估层（预留）

#### `predictions`

每次预测的 `purchase_pred` / `redeem_pred`，关联 `model_id`，`is_final=1` 标记最终提交版本。

#### `daily_evaluation`

每日相对误差、申购/赎回得分（0~10）、加权总分（申购×45% + 赎回×55%）。供误差分析模块使用。

---

### 3.5 数据质量层

#### `data_quality_report`

| 检查项 | 说明 |
|--------|------|
| `balance_equation` | 余额公式校验 |
| `yield_date_alignment` | 收益率日期与 daily_summary 对齐 |
| `consume_category_consistency` | consume=0 时 category 应为空 |

---

## 4. 视图

| 视图 | 用途 |
|------|------|
| `v_daily_purchase_redeem` | 申购/赎回/净流入/活跃用户趋势（数据探索模块） |
| `v_balance_violations` | 违反余额公式的记录（数据质量报告） |

---

## 5. 使用方式

### 前置条件

先跑完三个 PySpark 预处理脚本，生成 CSV 输出：

```powershell
python 01_data_exploration.py
python 02_feature_engineering.py
python 03_train_data_prepare.py
```

### 初始化数据库

```powershell
# 首次或重建
python database/init_db.py --force

# 指定路径
python database/init_db.py --db data/purchase_redemption.db --force
```

### 常用查询示例

```sql
-- 申购赎回趋势
SELECT * FROM v_daily_purchase_redeem ORDER BY report_date;

-- 验证集目标变量
SELECT report_date, total_purchase, total_redeem
FROM ml_val_data ORDER BY report_date;

-- 数据质量报告
SELECT * FROM data_quality_report;
```

---

## 6. 与项目模块的对应关系

| 项目要求 | 使用的表 |
|----------|----------|
| 数据预处理 & 一致性验证 | 原始层 + `data_quality_report` + `v_balance_violations` |
| 特征工程 | `daily_features`, `ml_train_data`, `ml_val_data` |
| 构建预测模型 | `ml_train_data`, `ml_val_data`, `dataset_split` |
| M1 数据探索 | `daily_summary`, `user_profile`, `mfd_*`, `v_daily_purchase_redeem` |
| M2 模型训练评估 | `model_registry`, `ml_train_data`, `ml_val_data` |
| M3 预测结果展示 | `predictions`, `comp_predict_template` |
| M4 误差分析 | `daily_evaluation`, `predictions` |

---

## 维护说明

- 修改表结构：编辑 `schema.sql`，然后 `python database/init_db.py --force` 重建
- 新增质量检查：在 `init_db.py` 的 `run_quality_checks()` 中添加
- 数据库文件 `data/*.db` 已加入 `.gitignore`，不入库
