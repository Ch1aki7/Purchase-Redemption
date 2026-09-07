"""统一路径、时间范围和实验参数。"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = ROOT / 'models'
OUTPUT_DIR = ROOT / 'output'
TRAIN_START, TRAIN_END = '2013-07-01', '2014-08-31'
FORECAST_START, FORECAST_END = '2014-09-01', '2014-09-30'
RANDOM_STATE = 42
LAG_FEATURES = (1, 2, 3, 7, 14, 21, 28, 30)
ROLLING_WINDOWS = (3, 7, 14, 30)
PURCHASE_WEIGHT, REDEEM_WEIGHT = .45, .55
TARGETS = ('purchase', 'redeem')
EPSILON = 1.0  # 原始金额单位；零值另行报告，MAPE 使用这个下限
FOLDS = (('2014-06-01','2014-06-30'), ('2014-07-01','2014-07-31'), ('2014-08-01','2014-08-31'))
