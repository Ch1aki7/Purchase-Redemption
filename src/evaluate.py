import numpy as np
import pandas as pd


PURCHASE_WEIGHT = 0.45
REDEEM_WEIGHT = 0.55
TAIL_WEIGHT = 0.15
OVER_20_WEIGHT = 0.03
OVER_30_WEIGHT = 0.05


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
    required = {"purchase_true", "purchase_pred", "redeem_true", "redeem_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"评估数据缺少字段：{sorted(missing)}")
    if df.empty:
        raise ValueError("评估数据不能为空")
    purchase_error = relative_error(df["purchase_true"], df["purchase_pred"])
    redeem_error = relative_error(df["redeem_true"], df["redeem_pred"])
    if not np.isfinite(purchase_error).all() or not np.isfinite(redeem_error).all():
        raise ValueError("相对误差包含 NaN 或无穷值")

    purchase_mape = float(purchase_error.mean())
    redeem_mape = float(redeem_error.mean())
    purchase_p90 = float(np.quantile(purchase_error, 0.90))
    redeem_p90 = float(np.quantile(redeem_error, 0.90))
    purchase_over_20 = purchase_error > 0.20
    redeem_over_20 = redeem_error > 0.20
    purchase_over_30 = purchase_error > 0.30
    redeem_over_30 = redeem_error > 0.30

    flow_mape = PURCHASE_WEIGHT * purchase_mape + REDEEM_WEIGHT * redeem_mape
    tail_p90 = PURCHASE_WEIGHT * purchase_p90 + REDEEM_WEIGHT * redeem_p90
    severe_20_ratio = (
        PURCHASE_WEIGHT * float(purchase_over_20.mean())
        + REDEEM_WEIGHT * float(redeem_over_20.mean())
    )
    severe_30_ratio = (
        PURCHASE_WEIGHT * float(purchase_over_30.mean())
        + REDEEM_WEIGHT * float(redeem_over_30.mean())
    )
    risk_adjusted_loss = (
        flow_mape
        + TAIL_WEIGHT * tail_p90
        + OVER_20_WEIGHT * severe_20_ratio
        + OVER_30_WEIGHT * severe_30_ratio
    )

    purchase_score = mock_score_from_error(purchase_error).mean()
    redeem_score = mock_score_from_error(redeem_error).mean()
    total_score = purchase_score * PURCHASE_WEIGHT + redeem_score * REDEEM_WEIGHT
    return {
        "sample_days": int(len(df)),
        "purchase_mape": purchase_mape,
        "redeem_mape": redeem_mape,
        "flow_mape": float(flow_mape),
        "purchase_median_ape": float(np.median(purchase_error)),
        "redeem_median_ape": float(np.median(redeem_error)),
        "purchase_p90_ape": purchase_p90,
        "redeem_p90_ape": redeem_p90,
        "tail_p90": float(tail_p90),
        "purchase_over_20_days": int(purchase_over_20.sum()),
        "redeem_over_20_days": int(redeem_over_20.sum()),
        "purchase_over_20_ratio": float(purchase_over_20.mean()),
        "redeem_over_20_ratio": float(redeem_over_20.mean()),
        "purchase_over_30_days": int(purchase_over_30.sum()),
        "redeem_over_30_days": int(redeem_over_30.sum()),
        "purchase_over_30_ratio": float(purchase_over_30.mean()),
        "redeem_over_30_ratio": float(redeem_over_30.mean()),
        "severe_20_ratio": float(severe_20_ratio),
        "severe_30_ratio": float(severe_30_ratio),
        "risk_adjusted_loss": float(risk_adjusted_loss),
        # 以下为旧页面和历史结果兼容字段，不作为新选型主指标。
        "purchase_score": float(purchase_score),
        "redeem_score": float(redeem_score),
        "total_score": float(total_score),
    }
