import numpy as np
import pandas as pd


def relative_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.where(y_true == 0, 1.0, y_true)
    return np.abs(y_pred - y_true) / denominator


def absolute_error(y_true, y_pred):
    return np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))


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


def build_daily_error_frame(df: pd.DataFrame) -> pd.DataFrame:
    """生成每日绝对误差、相对误差与模拟得分，便于可视化与高亮。"""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["purchase_abs_error"] = absolute_error(out["purchase_true"], out["purchase_pred"])
    out["redeem_abs_error"] = absolute_error(out["redeem_true"], out["redeem_pred"])
    out["purchase_rel_error"] = relative_error(out["purchase_true"], out["purchase_pred"])
    out["redeem_rel_error"] = relative_error(out["redeem_true"], out["redeem_pred"])
    out["purchase_score"] = mock_score_from_error(out["purchase_rel_error"])
    out["redeem_score"] = mock_score_from_error(out["redeem_rel_error"])
    out["daily_score"] = out["purchase_score"] * 0.45 + out["redeem_score"] * 0.55
    return out.sort_values("date").reset_index(drop=True)
