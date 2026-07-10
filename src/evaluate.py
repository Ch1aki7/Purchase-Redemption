import numpy as np
import pandas as pd


def relative_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.where(y_true == 0, 1.0, y_true)
    return np.abs(y_pred - y_true) / denominator


def mock_score_from_error(error):
    """赛题真实得分函数不公开，这里用线性函数做课程项目中的近似模拟。"""
    error = np.asarray(error, dtype=float)
    return np.maximum(0, 10 * (1 - error / 0.3))


def evaluate_prediction(df: pd.DataFrame) -> dict:
    purchase_error = relative_error(df["purchase_true"], df["purchase_pred"])
    redeem_error = relative_error(df["redeem_true"], df["redeem_pred"])
    purchase_score = mock_score_from_error(purchase_error).mean()
    redeem_score = mock_score_from_error(redeem_error).mean()
    total_score = purchase_score * 0.45 + redeem_score * 0.55
    return {
        "purchase_mape": float(purchase_error.mean()),
        "redeem_mape": float(redeem_error.mean()),
        "purchase_score": float(purchase_score),
        "redeem_score": float(redeem_score),
        "total_score": float(total_score),
    }
