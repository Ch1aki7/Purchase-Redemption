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
DIRECT_BACKTEST_PRED_PATH = OUTPUT_DIR / "direct_backtest_predictions.csv"
DIRECT_BACKTEST_METRICS_PATH = OUTPUT_DIR / "direct_backtest_metrics.csv"
DIRECT_CANDIDATE_PATH = OUTPUT_DIR / "tc_comp_predict_table_direct_candidate.csv"
RESIDUAL_CANDIDATE_PATH = OUTPUT_DIR / "tc_comp_predict_table_residual_candidate.csv"
SHAPE_CANDIDATE_PATH = OUTPUT_DIR / "tc_comp_predict_table_shape_candidate.csv"
SHAPE_BACKTEST_METRICS_PATH = OUTPUT_DIR / "shape_candidate_backtest_metrics.csv"
WEEKLY_BACKTEST_METRICS_PATH = OUTPUT_DIR / "weekly_forecast_backtest_metrics.csv"
WEEKLY_CANDIDATE_PATH = OUTPUT_DIR / "tc_comp_predict_table_weekly_candidate.csv"
CALIBRATION_METRICS_PATH = OUTPUT_DIR / "error_calibration_metrics.csv"
CALIBRATION_CANDIDATE_PATH = OUTPUT_DIR / "tc_comp_predict_table_calibrated_candidate.csv"
CALENDAR_METRICS_PATH = OUTPUT_DIR / "calendar_forecast_metrics.csv"
CALENDAR_CANDIDATE_PATH = OUTPUT_DIR / "tc_comp_predict_table_calendar_candidate.csv"
SUBMISSION_PATH = OUTPUT_DIR / "tc_comp_predict_table.csv"
SUBMISSION_WITH_HEADER_PATH = OUTPUT_DIR / "tc_comp_predict_table_with_header.csv"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
MODEL_PURCHASE_PATH = MODEL_DIR / "model_purchase.pkl"
MODEL_REDEEM_PATH = MODEL_DIR / "model_redeem.pkl"
