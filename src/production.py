"""Publish the leaderboard-validated ensemble as the single production result."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .calendar_forecast import _fold, _forecast_from_params
from .config import (
    BACKTEST_PRED_PATH,
    CALENDAR_METRICS_PATH,
    DAILY_FEATURES_PATH,
    EVENT_DISTANCE_METRICS_PATH,
    EVENT_DISTANCE_REDEEM_ONLY_PATH,
    MONTHLY_SHAPE_METRICS_PATH,
    SUBMISSION_PATH,
    SUBMISSION_WITH_HEADER_PATH,
    VALIDATION_PRED_PATH,
)
from .event_distance_forecast import (
    holt_winters_forecast,
    log_ensemble,
    ridge_forecast,
)
from .monthly_shape_forecast import predict_selected


COLUMNS = ["report_date", "purchase", "redeem"]
HOLDOUT_FOLD = "2014-08"


def _selected_row(path, target: str, accepted_only: bool = False) -> dict:
    metrics = pd.read_csv(path)
    rows = metrics[(metrics["row_type"] == "selected") & (metrics["target"] == target)]
    if accepted_only:
        rows = rows[rows["accepted"].astype(str).str.lower() == "true"]
    if len(rows) != 1:
        raise ValueError(f"{path} 中找不到唯一的 {target} 已选参数")
    return rows.iloc[0].to_dict()


def _validate_submission(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape != (30, 3) or list(frame.columns) != COLUMNS:
        raise ValueError("正式预测必须为30行3列：report_date、purchase、redeem")
    expected_dates = list(range(20140901, 20140931))
    if frame["report_date"].astype(int).tolist() != expected_dates:
        raise ValueError("正式预测日期必须完整覆盖2014年9月且按日期升序")
    values = frame[["purchase", "redeem"]].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("正式预测包含空值或非有限值")
    if (values <= 0).any().any():
        raise ValueError("正式预测金额必须为正数")
    result = frame.copy()
    result["report_date"] = result["report_date"].astype(np.int64)
    result[["purchase", "redeem"]] = np.rint(values).astype(np.int64)
    return result


def _build_matching_holdout() -> pd.DataFrame:
    """Recreate August with the same structure used by the production submission."""
    data = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    backtest = pd.read_csv(BACKTEST_PRED_PATH, parse_dates=["date"])
    baseline = backtest[backtest["strategy"] == "month_end_boost"].copy()
    _, test = _fold(data, baseline, HOLDOUT_FOLD)
    if test.empty:
        raise ValueError("滚动回测中缺少2014年8月结果")

    calendar_metrics = pd.read_csv(CALENDAR_METRICS_PATH)
    calendar_selected = calendar_metrics[calendar_metrics["accepted"].notna()]
    monthly_metrics = pd.read_csv(MONTHLY_SHAPE_METRICS_PATH)
    predictions = {}

    for target in ("purchase", "redeem"):
        calendar_rows = calendar_selected[calendar_selected["target"] == target]
        if len(calendar_rows) != 1:
            raise ValueError(f"日历模型中找不到唯一的 {target} 参数")
        reference = _forecast_from_params(
            data,
            test["date"],
            test[f"{target}_pred"],
            target,
            calendar_rows.iloc[0].to_dict(),
        )
        monthly_rows = monthly_metrics[
            (monthly_metrics["row_type"] == "selected")
            & (monthly_metrics["target"] == target)
        ]
        if len(monthly_rows) != 1:
            raise ValueError(f"月度形状模型中找不到唯一的 {target} 参数")
        monthly = monthly_rows.iloc[0].to_dict()
        accepted = str(monthly.get("accepted", "false")).lower() == "true"
        predictions[target] = (
            predict_selected(data[data["date"] < test["date"].min()], test["date"], target, reference, monthly)
            if accepted
            else reference
        )

    event = _selected_row(EVENT_DISTANCE_METRICS_PATH, "redeem")
    history = data[data["date"] < test["date"].min()]
    ridge_signal = ridge_forecast(history, test["date"], "redeem", json.loads(event["ridge_spec"]))
    hw_signal = holt_winters_forecast(history, test["date"], "redeem", json.loads(event["hw_spec"]))
    predictions["redeem"] = log_ensemble(
        predictions["redeem"],
        ridge_signal,
        hw_signal,
        float(event["ridge_weight"]),
        float(event["hw_weight"]),
        test["date"],
        float(event["event_scale"]),
    )

    return pd.DataFrame({
        "date": test["date"].to_numpy(),
        "purchase_true": test["purchase_true"].astype(np.int64).to_numpy(),
        "purchase_pred": np.rint(predictions["purchase"]).astype(np.int64),
        "redeem_true": test["redeem_true"].astype(np.int64).to_numpy(),
        "redeem_pred": np.rint(predictions["redeem"]).astype(np.int64),
    })


def main():
    production = pd.read_csv(
        EVENT_DISTANCE_REDEEM_ONLY_PATH,
        header=None,
        names=COLUMNS,
    )
    production = _validate_submission(production)
    production.to_csv(SUBMISSION_PATH, index=False, header=False, encoding="utf-8")
    production.to_csv(SUBMISSION_WITH_HEADER_PATH, index=False, encoding="utf-8-sig")

    holdout = _build_matching_holdout()
    holdout.to_csv(VALIDATION_PRED_PATH, index=False, encoding="utf-8-sig")
    print(f"正式提交文件：{SUBMISSION_PATH}")
    print(f"前端展示文件：{SUBMISSION_WITH_HEADER_PATH}")
    print(f"匹配结构的8月验证：{VALIDATION_PRED_PATH}")


if __name__ == "__main__":
    main()
