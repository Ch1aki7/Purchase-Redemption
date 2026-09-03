from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Purchase Redemption Data"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = ROOT_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DAILY_FEATURES_PATH = OUTPUT_DIR / "daily_features.csv"
DATA_QUALITY_REPORT_PATH = OUTPUT_DIR / "data_quality_report.csv"
VALIDATION_PRED_PATH = OUTPUT_DIR / "validation_prediction.csv"
BACKTEST_PRED_PATH = OUTPUT_DIR / "rolling_backtest_predictions.csv"
BACKTEST_METRICS_PATH = OUTPUT_DIR / "rolling_backtest_metrics.csv"
SUBMISSION_PATH = OUTPUT_DIR / "tc_comp_predict_table.csv"
SUBMISSION_WITH_HEADER_PATH = OUTPUT_DIR / "tc_comp_predict_table_with_header.csv"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
MODEL_PURCHASE_PATH = MODEL_DIR / "model_purchase.pkl"
MODEL_REDEEM_PATH = MODEL_DIR / "model_redeem.pkl"

# 多月份滚动回测选出的默认周期融合参数。
SEASONAL_WEEKDAY_LOOKBACK = 12
PURCHASE_MODEL_WEIGHT = 1.0
REDEEM_MODEL_WEIGHT = 0.5
DEFAULT_MODEL_NAME = "xgboost"
DEFAULT_MODEL_PRESET = "shallow"
DEFAULT_N_SEEDS = 3
DEFAULT_XGBOOST_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 2,
    "min_child_weight": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
}
