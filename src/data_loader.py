from pathlib import Path
import pandas as pd
from .config import DATA_DIR


def read_csv_safely(path: Path) -> pd.DataFrame:
    """兼容 utf-8、gbk 等常见编码读取 CSV。"""
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"读取文件失败：{path}，最后错误：{last_error}")


def find_csv(keyword: str) -> Path:
    """在数据目录下按关键词查找 CSV/TXT 文件。"""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"数据目录不存在：{DATA_DIR}")
    candidates = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.txt"))
    keyword_lower = keyword.lower()
    matched = [p for p in candidates if keyword_lower in p.name.lower()]
    if not matched:
        names = "\n".join(p.name for p in candidates) or "（目录为空）"
        raise FileNotFoundError(
            f"在数据目录中找不到包含关键词 '{keyword}' 的 CSV/TXT 文件。\n"
            f"当前数据目录文件：\n{names}"
        )
    return matched[0]


def load_all_tables():
    """读取四张原始表。"""
    user_profile = read_csv_safely(find_csv("user_profile"))
    user_balance = read_csv_safely(find_csv("user_balance"))
    share_interest = read_csv_safely(find_csv("share_interest"))
    bank_shibor = read_csv_safely(find_csv("bank_shibor"))
    return user_profile, user_balance, share_interest, bank_shibor


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去除列名前后空格。"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df
