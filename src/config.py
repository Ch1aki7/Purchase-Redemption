from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Purchase Redemption Data"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = ROOT_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DAILY_FEATURES_PATH = OUTPUT_DIR / "daily_features.csv"
VALIDATION_PRED_PATH = OUTPUT_DIR / "validation_prediction.csv"
BACKTEST_PRED_PATH = OUTPUT_DIR / "rolling_backtest_predictions.csv"
BACKTEST_METRICS_PATH = OUTPUT_DIR / "rolling_backtest_metrics.csv"
SUBMISSION_PATH = OUTPUT_DIR / "tc_comp_predict_table.csv"
SUBMISSION_WITH_HEADER_PATH = OUTPUT_DIR / "tc_comp_predict_table_with_header.csv"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
MODEL_PURCHASE_PATH = MODEL_DIR / "model_purchase.pkl"
MODEL_REDEEM_PATH = MODEL_DIR / "model_redeem.pkl"
