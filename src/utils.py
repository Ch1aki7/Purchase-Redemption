"""日志与输出辅助函数。"""
import json
import logging
import numpy as np
from .config import OUTPUT_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RAW_DATA_DIR, RANDOM_STATE

def setup_logging():
    """创建输出目录，固定随机种子并配置日志。"""
    for p in (OUTPUT_DIR, PROCESSED_DATA_DIR, MODEL_DIR / 'purchase', MODEL_DIR / 'redeem', RAW_DATA_DIR):
        p.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / 'pipeline.log', encoding='utf8')])

def save_json(value, path):
    """写出可读 JSON。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf8')
