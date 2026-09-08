"""Non-recursive calendar/trend Ridge forecasts with an untouched holdout month."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import (
    BACKTEST_PRED_PATH,
    CALENDAR_CANDIDATE_PATH,
    CALENDAR_METRICS_PATH,
    DAILY_FEATURES_PATH,
    SUBMISSION_PATH,
)
from .predict import HOLIDAYS_2014


TUNE_FOLDS = ("2014-04", "2014-05", "2014-06", "2014-07")
HOLDOUT_FOLD = "2014-08"
WINDOWS = (120, 180, 240, 365)
LOG_SPACES = (False, True)
TREND_DEGREES = (1, 2)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
HALF_LIVES = (60.0, 120.0, 240.0, 10000.0)
BLEND_WEIGHTS = (0.10, 0.25, 0.50, 0.75, 1.0)
ORIGIN = pd.Timestamp("2013-07-01")


def calendar_matrix(dates, trend_degree: int) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    trend = ((dates - ORIGIN).days.to_numpy(dtype=float) / 365.0)
    weekday = dates.dayofweek.to_numpy()
    day = dates.day.to_numpy(dtype=float)
    month = dates.month.to_numpy(dtype=float)
    columns = [
        trend,
        *[(weekday == value).astype(float) for value in range(7)],
        np.sin(2 * np.pi * day / 31.0),
        np.cos(2 * np.pi * day / 31.0),
        np.sin(4 * np.pi * day / 31.0),
        np.cos(4 * np.pi * day / 31.0),
        np.sin(2 * np.pi * month / 12.0),
        np.cos(2 * np.pi * month / 12.0),
        (day <= 3).astype(float),
        (day >= dates.days_in_month.to_numpy() - 2).astype(float),
        np.asarray([date.strftime("%Y-%m-%d") in HOLIDAYS_2014 for date in dates], dtype=float),
    ]
    columns.extend([(weekday == value).astype(float) * trend for value in range(7)])
    if trend_degree == 2:
        columns.append(trend ** 2)
    return np.column_stack(columns)


def calendar_forecast(
    history: pd.DataFrame,
    forecast_dates,
    target: str,
    window: int,
    log_space: bool,
    trend_degree: int,
    ridge_alpha: float,
    half_life: float,
) -> np.ndarray:
    train = history.sort_values("date").tail(window)
    X = calendar_matrix(train["date"], trend_degree)
    X_future = calendar_matrix(forecast_dates, trend_degree)
    y = train[target].to_numpy(dtype=float)
    if log_space:
        y = np.log1p(y)
    age = (train["date"].max() - train["date"]).dt.days.to_numpy(dtype=float)
    sample_weight = np.exp(-np.log(2.0) * age / half_life)
    model = Ridge(alpha=ridge_alpha)
    model.fit(X, y, sample_weight=sample_weight)
    prediction = model.predict(X_future)
    if log_space:
        prediction = np.expm1(prediction)
    return np.maximum(0.0, prediction)


def _blend(baseline, calendar, weight: float):
    baseline = np.maximum(np.asarray(baseline, dtype=float), 1.0)
    calendar = np.maximum(np.asarray(calendar, dtype=float), 1.0)
    return np.exp((1.0 - weight) * np.log(baseline) + weight * np.log(calendar))


def _score(y_true, y_pred):
    error = np.abs(np.asarray(y_pred) - np.asarray(y_true)) / np.maximum(
        np.asarray(y_true, dtype=float), 1.0
    )
    return float(np.maximum(0.0, 10.0 * (1.0 - error / 0.3)).mean())


def _fold(data, baseline, fold: str):
    period = pd.Period(fold, freq="M")
    history = data[data["date"] < period.start_time]
    test = baseline[baseline["fold"].astype(str) == fold].sort_values("date")
    return history, test


def search_target(data: pd.DataFrame, baseline: pd.DataFrame, target: str):
    rows = []
    grid = itertools.product(
        WINDOWS, LOG_SPACES, TREND_DEGREES, RIDGE_ALPHAS, HALF_LIVES, BLEND_WEIGHTS
    )
    for window, log_space, degree, ridge_alpha, half_life, blend_weight in grid:
        deltas = []
        scores = []
        for fold in TUNE_FOLDS:
            history, test = _fold(data, baseline, fold)
            raw = calendar_forecast(
                history, test["date"], target, window, log_space,
                degree, ridge_alpha, half_life,
            )
            prediction = _blend(test[f"{target}_pred"], raw, blend_weight)
            score = _score(test[f"{target}_true"], prediction)
            base_score = _score(test[f"{target}_true"], test[f"{target}_pred"])
            scores.append(score)
            deltas.append(score - base_score)
        rows.append({
            "target": target,
            "window": window,
            "log_space": log_space,
            "trend_degree": degree,
            "ridge_alpha": ridge_alpha,
            "half_life": half_life,
            "blend_weight": blend_weight,
            "tune_score": float(np.mean(scores)),
            "tune_delta": float(np.mean(deltas)),
            "winning_folds": int(sum(delta > 0 for delta in deltas)),
            "worst_fold_delta": float(min(deltas)),
        })
    results = pd.DataFrame(rows).sort_values(
        ["tune_score", "winning_folds", "worst_fold_delta"], ascending=False
    )
    stable = results[
        (results["tune_delta"] > 0)
        & (results["winning_folds"] >= 3)
        & (results["worst_fold_delta"] >= -0.10)
    ]
    return (stable if not stable.empty else results).iloc[0].to_dict(), results


def _forecast_from_params(data, dates, baseline_values, target, params):
    history = data[data["date"] < pd.DatetimeIndex(dates).min()]
    raw = calendar_forecast(
        history, dates, target, int(params["window"]), bool(params["log_space"]),
        int(params["trend_degree"]), float(params["ridge_alpha"]),
        float(params["half_life"]),
    )
    return _blend(baseline_values, raw, float(params["blend_weight"]))


def main():
    data = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    predictions = pd.read_csv(BACKTEST_PRED_PATH, parse_dates=["date"])
    baseline = predictions[predictions["strategy"] == "month_end_boost"].copy()
    selected = {}
    result_frames = []
    holdout_rows = []

    for target in ("purchase", "redeem"):
        params, results = search_target(data, baseline, target)
        selected[target] = params
        result_frames.append(results)
        _, test = _fold(data, baseline, HOLDOUT_FOLD)
        holdout_prediction = _forecast_from_params(
            data, test["date"], test[f"{target}_pred"], target, params
        )
        delta = (
            _score(test[f"{target}_true"], holdout_prediction)
            - _score(test[f"{target}_true"], test[f"{target}_pred"])
        )
        holdout_rows.append({
            **params,
            "holdout_delta": delta,
            "accepted": bool(delta > 0),
        })

    pd.concat(result_frames + [pd.DataFrame(holdout_rows)], ignore_index=True).to_csv(
        CALENDAR_METRICS_PATH, index=False, encoding="utf-8-sig"
    )
    submission = pd.read_csv(
        SUBMISSION_PATH, header=None,
        names=["report_date", "purchase", "redeem"],
    )
    dates = pd.to_datetime(submission["report_date"].astype(str), format="%Y%m%d")
    candidate = submission.copy()
    for params in holdout_rows:
        target = params["target"]
        if params["accepted"]:
            candidate[target] = np.maximum(0, np.rint(_forecast_from_params(
                data, dates, submission[target], target, params
            ))).astype(np.int64)
    candidate.to_csv(CALENDAR_CANDIDATE_PATH, index=False, header=False, encoding="utf-8")

    print("4-7月选参与8月独立门禁：")
    print(pd.DataFrame(holdout_rows)[[
        "target", "window", "log_space", "trend_degree", "ridge_alpha",
        "half_life", "blend_weight", "tune_delta", "winning_folds",
        "worst_fold_delta", "holdout_delta", "accepted",
    ]].to_string(index=False))
    print(f"候选文件：{CALENDAR_CANDIDATE_PATH}")


if __name__ == "__main__":
    main()
