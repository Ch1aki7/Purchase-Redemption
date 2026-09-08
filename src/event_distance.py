"""Production forecaster adapted to the current data and output contracts.

The frozen September forecast is the reference submission.  The production
Ridge and weekly Holt-Winters signals are trained from C2's daily aggregate,
then blended with that reference.  Finally, the target-isolation step from
target-isolation step writes one submission for each target.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import (
    EVENT_DISTANCE_CANDIDATE_PATH,
    EVENT_DISTANCE_MANIFEST_PATH,
    EVENT_DISTANCE_PURCHASE_ONLY_PATH,
    EVENT_DISTANCE_REDEEM_ONLY_PATH,
    PREDICTION_NO_HEADER_PATH,
    PROCESSED_DATA_DIR,
)


SUBMISSION_COLUMNS = ["report_date", "purchase", "redeem"]
HOLIDAY_PERIODS = (
    ("2013-09-19", "2013-09-21"),
    ("2013-10-01", "2013-10-07"),
    ("2014-01-01", "2014-01-01"),
    ("2014-01-31", "2014-02-06"),
    ("2014-04-05", "2014-04-07"),
    ("2014-05-01", "2014-05-03"),
    ("2014-05-31", "2014-06-02"),
    ("2014-09-06", "2014-09-08"),
    ("2014-10-01", "2014-10-07"),
)
SPECIAL_WORKDAYS = {
    "2013-09-22", "2013-09-29", "2013-10-12",
    "2014-01-26", "2014-02-08", "2014-05-04",
    "2014-09-28", "2014-10-11",
}

DEFAULT_PARAMS = {
    "ridge_window": 365,
    "ridge_alpha": 10.0,
    "ridge_half_life": 120.0,
    "ridge_log_space": True,
    "hw_window": 224,
    "hw_alpha": 0.25,
    "hw_beta": 0.05,
    "hw_gamma": 0.30,
    "hw_damping": 0.95,
    "hw_log_space": True,
    "purchase_ridge_weight": 0.25,
    "purchase_hw_weight": 0.25,
    "redeem_ridge_weight": 0.25,
    "redeem_hw_weight": 0.25,
    "event_scale": 0.50,
}


def load_submission(path: str | Path) -> pd.DataFrame:
    """Load and strictly validate the competition's 30-row contract."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"缺少基线提交文件：{path}。请先运行 python run_all.py")
    frame = pd.read_csv(path, header=None, names=SUBMISSION_COLUMNS)
    expected_dates = list(range(20140901, 20140931))
    if frame.shape != (30, 3):
        raise ValueError(f"{path} 不是30行3列")
    if frame["report_date"].tolist() != expected_dates:
        raise ValueError(f"{path} 日期不完整或顺序错误")
    if not np.isfinite(frame[["purchase", "redeem"]].to_numpy(float)).all():
        raise ValueError(f"{path} 存在非有限预测值")
    if not (frame[["purchase", "redeem"]] > 0).all().all():
        raise ValueError(f"{path} 存在非正预测值")
    return frame


def normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Map either a current daily frame or an already-normalized frame to model names."""
    data = frame.reset_index() if "date" not in frame.columns else frame.copy()
    data = data.rename(columns={"total_purchase": "purchase", "total_redeem": "redeem"})
    missing = {"date", "purchase", "redeem"} - set(data.columns)
    if missing:
        raise ValueError(f"日聚合数据缺少字段：{sorted(missing)}")
    data = data[["date", "purchase", "redeem"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").drop_duplicates("date", keep="last")
    if data.empty or data[["purchase", "redeem"]].isna().any().any():
        raise ValueError("日聚合数据为空或目标列含缺失值")
    return data


def event_matrix(dates) -> np.ndarray:
    """Calendar and event-distance design matrix used by the production model."""
    dates = pd.DatetimeIndex(dates)
    starts = pd.DatetimeIndex([start for start, _ in HOLIDAY_PERIODS])
    ends = pd.DatetimeIndex([end for _, end in HOLIDAY_PERIODS])
    special_work = pd.DatetimeIndex(sorted(SPECIAL_WORKDAYS))
    weekday = dates.dayofweek.to_numpy()
    day = dates.day.to_numpy()
    days_in_month = dates.days_in_month.to_numpy()
    trend = np.asarray((dates - pd.Timestamp("2013-07-01")).days, dtype=float) / 365.0
    is_holiday = np.zeros(len(dates), dtype=float)
    position = np.zeros(len(dates), dtype=int)
    holiday_length = np.zeros(len(dates), dtype=int)
    for start_text, end_text in HOLIDAY_PERIODS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        mask = (dates >= start) & (dates <= end)
        is_holiday[mask] = 1.0
        position[mask] = np.asarray((dates[mask] - start).days) + 1
        holiday_length[mask] = (end - start).days + 1
    is_special_work = dates.isin(special_work).astype(float)
    is_work = (((weekday < 5) & (is_holiday == 0)) | (is_special_work == 1)).astype(float)
    to_start = np.asarray([
        min([(start - date).days for start in starts if start >= date] or [30])
        for date in dates
    ])
    from_end = np.asarray([
        min([(date - end).days for end in ends if end <= date] or [30])
        for date in dates
    ])
    progress = (day - 1) / np.maximum(days_in_month - 1, 1)
    columns = [
        trend, trend ** 2,
        *[(weekday == value).astype(float) for value in range(7)],
        *[(day == value).astype(float) for value in range(1, 32)],
        *[(dates.month == value).astype(float) for value in range(1, 13)],
        *[((day - 1) // 7 == value).astype(float) for value in range(5)],
        np.sin(2 * np.pi * progress), np.cos(2 * np.pi * progress),
        np.sin(4 * np.pi * progress), np.cos(4 * np.pi * progress),
        is_holiday, is_work, is_special_work,
        *[(position == value).astype(float) for value in range(1, 8)],
        ((is_holiday == 1) & (position == holiday_length)).astype(float),
        *[(to_start == value).astype(float) for value in range(1, 8)],
        *[(from_end == value).astype(float) for value in range(1, 8)],
        (day <= 3).astype(float), (day >= days_in_month - 2).astype(float),
        *[(day == value).astype(float) for value in (5, 10, 15, 20, 25)],
        *[(weekday == value).astype(float) * is_work for value in range(7)],
    ]
    return np.column_stack(columns)


def ridge_forecast(history, dates, target: str, params: dict) -> np.ndarray:
    train = history.sort_values("date").tail(int(params["ridge_window"]))
    if len(train) < 30:
        raise ValueError("正式模型 Ridge 训练历史不足30天")
    y = train[target].to_numpy(float)
    if params["ridge_log_space"]:
        y = np.log1p(y)
    age = (train["date"].max() - train["date"]).dt.days.to_numpy(float)
    weights = np.exp(-np.log(2.0) * age / float(params["ridge_half_life"]))
    model = Ridge(alpha=float(params["ridge_alpha"]))
    model.fit(event_matrix(train["date"]), y, sample_weight=weights)
    prediction = model.predict(event_matrix(dates))
    if params["ridge_log_space"]:
        prediction = np.expm1(prediction)
    bounds = np.percentile(train[target], [1, 99])
    return np.clip(prediction, max(1.0, bounds[0] * 0.5), bounds[1] * 1.5)


def holt_winters_forecast(history, dates, target: str, params: dict) -> np.ndarray:
    raw = history.sort_values("date")[target].tail(int(params["hw_window"])).to_numpy(float)
    if len(raw) < 14:
        raise ValueError("Holt-Winters 训练历史不足14天")
    values = np.log1p(raw) if params["hw_log_space"] else raw.copy()
    season = 7
    level = float(np.mean(values[:season]))
    trend = float((np.mean(values[season:2 * season]) - level) / season)
    seasonal = np.asarray([
        np.mean(values[pos::season] - np.mean(values)) for pos in range(season)
    ], dtype=float)
    alpha = float(params["hw_alpha"])
    beta = float(params["hw_beta"])
    gamma = float(params["hw_gamma"])
    damping = float(params["hw_damping"])
    for index, value in enumerate(values):
        pos = index % season
        previous_level = level
        level = alpha * (value - seasonal[pos]) + (1 - alpha) * (level + damping * trend)
        trend = beta * (level - previous_level) + (1 - beta) * damping * trend
        seasonal[pos] = gamma * (value - level) + (1 - gamma) * seasonal[pos]
    predictions = []
    for horizon in range(1, len(pd.DatetimeIndex(dates)) + 1):
        damped_steps = sum(damping ** step for step in range(1, horizon + 1))
        pos = (len(values) + horizon - 1) % season
        predictions.append(level + damped_steps * trend + seasonal[pos])
    prediction = np.asarray(predictions)
    if params["hw_log_space"]:
        prediction = np.expm1(prediction)
    bounds = np.percentile(raw, [1, 99])
    return np.clip(prediction, max(1.0, bounds[0] * 0.5), bounds[1] * 1.5)


def event_window_mask(dates) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    mask = dates.isin(pd.DatetimeIndex(sorted(SPECIAL_WORKDAYS)))
    for start_text, end_text in HOLIDAY_PERIODS:
        start = pd.Timestamp(start_text) - pd.Timedelta(days=1)
        end = pd.Timestamp(end_text) + pd.Timedelta(days=1)
        mask |= (dates >= start) & (dates <= end)
    return np.asarray(mask, dtype=bool)


def log_ensemble(reference, ridge_signal, hw_signal, ridge_weight, hw_weight,
                 dates, event_scale: float) -> np.ndarray:
    if ridge_weight < 0 or hw_weight < 0 or ridge_weight + hw_weight > 1:
        raise ValueError("Ridge 与 Holt-Winters 权重必须非负且总和不超过1")
    reference = np.asarray(reference, dtype=float)
    ridge_weights = np.full(len(reference), float(ridge_weight))
    hw_weights = np.full(len(reference), float(hw_weight))
    if event_scale < 1.0:
        mask = event_window_mask(dates)
        ridge_weights[mask] *= event_scale
        hw_weights[mask] *= event_scale
    reference_weights = 1.0 - ridge_weights - hw_weights
    return np.exp(
        reference_weights * np.log(np.maximum(reference, 1.0))
        + ridge_weights * np.log(np.maximum(ridge_signal, 1.0))
        + hw_weights * np.log(np.maximum(hw_signal, 1.0))
    )


def create_target_isolated_submissions(
    baseline_path=None,
    candidate_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target-isolation logic: replace exactly one target at a time."""
    baseline_path = PREDICTION_NO_HEADER_PATH if baseline_path is None else baseline_path
    candidate_path = EVENT_DISTANCE_CANDIDATE_PATH if candidate_path is None else candidate_path
    baseline = load_submission(baseline_path)
    candidate = load_submission(candidate_path)
    purchase_only = baseline.copy()
    purchase_only["purchase"] = candidate["purchase"]
    redeem_only = baseline.copy()
    redeem_only["redeem"] = candidate["redeem"]
    purchase_only.to_csv(EVENT_DISTANCE_PURCHASE_ONLY_PATH, index=False, header=False, encoding="utf-8")
    redeem_only.to_csv(EVENT_DISTANCE_REDEEM_ONLY_PATH, index=False, header=False, encoding="utf-8")
    return purchase_only, redeem_only


def train_and_export(history: pd.DataFrame | None = None, params: dict | None = None) -> dict:
    """Low-level manual blend helper retained for focused tests and experiments.

    The application and module command use ``c3_full_pipeline.run_full_pipeline``
    so production outputs always come from the complete formal-model selection.
    """
    resolved = {**DEFAULT_PARAMS, **(params or {})}
    for target in ("purchase", "redeem"):
        total = resolved[f"{target}_ridge_weight"] + resolved[f"{target}_hw_weight"]
        if total > 1.0:
            raise ValueError(f"{target} 的 Ridge 与 Holt-Winters 权重之和不能超过1")
    if not 0.0 <= float(resolved["event_scale"]) <= 1.0:
        raise ValueError("事件窗口缩放必须在0到1之间")

    if history is None:
        history = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    data = normalize_history(history)
    baseline = load_submission(PREDICTION_NO_HEADER_PATH)
    dates = pd.to_datetime(baseline["report_date"].astype(str), format="%Y%m%d")
    train = data[data["date"] < dates.min()]
    if train.empty:
        raise ValueError("预测起点之前没有可用训练数据")

    candidate = baseline.copy()
    diagnostics = []
    for target in ("purchase", "redeem"):
        ridge_signal = ridge_forecast(train, dates, target, resolved)
        hw_signal = holt_winters_forecast(train, dates, target, resolved)
        blended = log_ensemble(
            baseline[target], ridge_signal, hw_signal,
            resolved[f"{target}_ridge_weight"], resolved[f"{target}_hw_weight"],
            dates, float(resolved["event_scale"]),
        )
        candidate[target] = np.maximum(1, np.rint(blended)).astype(np.int64)
        diagnostics.append({
            "target": target,
            "reference_mean": float(baseline[target].mean()),
            "ridge_mean": float(np.mean(ridge_signal)),
            "holt_winters_mean": float(np.mean(hw_signal)),
            "candidate_mean": float(candidate[target].mean()),
            "changed_days": int((candidate[target] != baseline[target]).sum()),
        })

    EVENT_DISTANCE_CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(EVENT_DISTANCE_CANDIDATE_PATH, index=False, header=False, encoding="utf-8")
    purchase_only, redeem_only = create_target_isolated_submissions()
    manifest = {
        "baseline": str(PREDICTION_NO_HEADER_PATH),
        "candidate": str(EVENT_DISTANCE_CANDIDATE_PATH),
        "purchase_only": str(EVENT_DISTANCE_PURCHASE_ONLY_PATH),
        "redeem_only": str(EVENT_DISTANCE_REDEEM_ONLY_PATH),
        "train_start": train["date"].min().strftime("%Y-%m-%d"),
        "train_end": train["date"].max().strftime("%Y-%m-%d"),
        "forecast_start": dates.min().strftime("%Y-%m-%d"),
        "forecast_end": dates.max().strftime("%Y-%m-%d"),
        "params": resolved,
        "diagnostics": diagnostics,
    }
    EVENT_DISTANCE_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "candidate": candidate,
        "purchase_only": purchase_only,
        "redeem_only": redeem_only,
        "manifest": manifest,
        "diagnostics": pd.DataFrame(diagnostics),
    }


if __name__ == "__main__":
    from .c3_full_pipeline import run_full_pipeline
    result = run_full_pipeline(callback=print)
    print(json.dumps(result["manifest"]["outputs"], ensure_ascii=False, indent=2))
