-- 资金流入流出预测系统 - SQLite 数据库结构
-- 分层：原始数据 → 加工数据 → 建模数据 → 预测与评估

PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. 原始数据层（四张竞赛表 + 提交模板）
-- ============================================================

CREATE TABLE IF NOT EXISTS user_profile (
    user_id       INTEGER PRIMARY KEY,
    sex           INTEGER NOT NULL,          -- 0=女, 1=男
    city          INTEGER NOT NULL,
    constellation TEXT
);

CREATE TABLE IF NOT EXISTS user_balance (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL,
    report_date         INTEGER NOT NULL,    -- YYYYMMDD
    t_balance           INTEGER NOT NULL,    -- 今日余额（分）
    y_balance           INTEGER NOT NULL,    -- 昨日余额（分）
    total_purchase_amt  INTEGER NOT NULL DEFAULT 0,
    direct_purchase_amt INTEGER NOT NULL DEFAULT 0,
    purchase_bal_amt    INTEGER NOT NULL DEFAULT 0,
    purchase_bank_amt   INTEGER NOT NULL DEFAULT 0,
    total_redeem_amt    INTEGER NOT NULL DEFAULT 0,
    consume_amt         INTEGER NOT NULL DEFAULT 0,
    transfer_amt        INTEGER NOT NULL DEFAULT 0,
    tftobal_amt         INTEGER NOT NULL DEFAULT 0,
    tftocard_amt        INTEGER NOT NULL DEFAULT 0,
    share_amt           INTEGER NOT NULL DEFAULT 0,
    category1           INTEGER,
    category2           INTEGER,
    category3           INTEGER,
    category4           INTEGER,
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id)
);

CREATE INDEX IF NOT EXISTS idx_user_balance_date ON user_balance(report_date);
CREATE INDEX IF NOT EXISTS idx_user_balance_user ON user_balance(user_id);
CREATE INDEX IF NOT EXISTS idx_user_balance_user_date ON user_balance(user_id, report_date);

CREATE TABLE IF NOT EXISTS mfd_day_share_interest (
    mfd_date          INTEGER PRIMARY KEY,   -- YYYYMMDD
    mfd_daily_yield   REAL NOT NULL,         -- 万份收益
    mfd_7daily_yield  REAL NOT NULL          -- 七日年化（%）
);

CREATE TABLE IF NOT EXISTS mfd_bank_shibor (
    mfd_date      INTEGER PRIMARY KEY,
    interest_o_n  REAL,
    interest_1_w  REAL,
    interest_2_w  REAL,
    interest_1_m  REAL,
    interest_3_m  REAL,
    interest_6_m  REAL,
    interest_9_m  REAL,
    interest_1_y  REAL
);

CREATE TABLE IF NOT EXISTS comp_predict_template (
    report_date INTEGER PRIMARY KEY,
    purchase    INTEGER NOT NULL,
    redeem      INTEGER NOT NULL
);

-- ============================================================
-- 2. 加工数据层（探索 + 特征工程输出）
-- ============================================================

CREATE TABLE IF NOT EXISTS daily_summary (
    report_date           INTEGER PRIMARY KEY,
    active_users          INTEGER,
    total_purchase        INTEGER,
    total_redeem          INTEGER,
    direct_purchase       INTEGER,
    purchase_bal          INTEGER,
    purchase_bank         INTEGER,
    total_share           INTEGER,
    total_consume         INTEGER,
    total_transfer        INTEGER,
    total_tftobal         INTEGER,
    total_tftocard        INTEGER,
    cat1_total            INTEGER,
    cat2_total            INTEGER,
    cat3_total            INTEGER,
    cat4_total            INTEGER,
    avg_balance           REAL,
    avg_yesterday_balance REAL
);

-- 完整特征表：列结构与 output/daily_features.csv 对齐，由 init_db.py 动态建表/导入
-- 此处仅声明核心字段，完整列在导入脚本中处理

CREATE TABLE IF NOT EXISTS daily_features_meta (
    feature_name TEXT PRIMARY KEY,
    feature_group TEXT,          -- time / lag / rolling / yield / shibor / profile ...
    description  TEXT
);

-- ============================================================
-- 3. 建模数据层
-- ============================================================

CREATE TABLE IF NOT EXISTS dataset_split (
    report_date INTEGER PRIMARY KEY,
    split_type  TEXT NOT NULL CHECK(split_type IN ('train', 'val', 'predict')),
    note        TEXT
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name   TEXT NOT NULL,
    target       TEXT NOT NULL CHECK(target IN ('purchase', 'redeem', 'both')),
    algorithm    TEXT NOT NULL,       -- lightgbm / xgboost / arima / ...
    params_json  TEXT,
    train_score  REAL,
    val_score    REAL,
    feature_count INTEGER,
    artifact_path TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- 4. 预测与评估层
-- ============================================================

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      INTEGER,
    report_date   INTEGER NOT NULL,
    purchase_pred INTEGER NOT NULL,
    redeem_pred   INTEGER NOT NULL,
    is_final      INTEGER NOT NULL DEFAULT 0,  -- 1=最终提交版本
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(report_date);

CREATE TABLE IF NOT EXISTS daily_evaluation (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id       INTEGER NOT NULL,
    report_date         INTEGER NOT NULL,
    purchase_actual     INTEGER,
    redeem_actual       INTEGER,
    purchase_pred       INTEGER NOT NULL,
    redeem_pred         INTEGER NOT NULL,
    purchase_rel_error  REAL,
    redeem_rel_error    REAL,
    purchase_score      REAL,        -- 0~10
    redeem_score        REAL,
    weighted_score      REAL,        -- purchase*0.45 + redeem*0.55
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
);

CREATE INDEX IF NOT EXISTS idx_daily_eval_date ON daily_evaluation(report_date);

-- ============================================================
-- 5. 数据质量层
-- ============================================================

CREATE TABLE IF NOT EXISTS data_quality_report (
    check_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    check_name  TEXT NOT NULL,
    check_result TEXT NOT NULL,      -- pass / warn / fail
    metric_value REAL,
    detail      TEXT,
    checked_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- 6. 常用视图
-- ============================================================

CREATE VIEW IF NOT EXISTS v_daily_purchase_redeem AS
SELECT
    report_date,
    total_purchase,
    total_redeem,
    total_purchase - total_redeem AS net_flow,
    active_users
FROM daily_summary
ORDER BY report_date;

CREATE VIEW IF NOT EXISTS v_balance_violations AS
SELECT
    user_id,
    report_date,
    y_balance,
    total_purchase_amt,
    total_redeem_amt,
    t_balance,
    (y_balance + total_purchase_amt - total_redeem_amt) AS expected_balance,
    t_balance - (y_balance + total_purchase_amt - total_redeem_amt) AS balance_diff
FROM user_balance
WHERE t_balance != (y_balance + total_purchase_amt - total_redeem_amt);
