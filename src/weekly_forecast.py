"""Direct weekly-sequence forecasting with a strict tune/holdout split.

Each weekday is modelled as its own weekly series. Parameters are selected only
on April-July 2014; August is reported once as an untouched holdout. The module
never overwrites the proven baseline submission.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .config import (
    BACKTEST_PRED_PATH,
    DAILY_FEATURES_PATH,
    SUBMISSION_PATH,
    WEEKLY_BACKTEST_METRICS_PATH,
    WEEKLY_CANDIDATE_PATH,
)


TUNE_FOLDS = ("2014-04", "2014-05", "2014-06", "2014-07")
HOLDOUT_FOLD = "2014-08"
WINDOWS = (8, 12, 16, 24, 32)
DECAYS = (0.90, 0.95, 1.0)
TREND_STRENGTHS = (0.0, 0.25, 0.5, 1.0)
LOG_SPACES = (False, True)
BLEND_WEIGHTS = (0.10, 0.25, 0.50, 0.75, 1.0)


def _weighted_line(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    x_mean = float(np.average(x, weights=weights))
    y_mean = float(np.average(y, weights=weights))
    denominator = float(np.sum(weights * (x - x_mean) ** 2))
    slope = (
        float(np.sum(weights * (x - x_mean) * (y - y_mean))) / denominator
        if denominator > 0 else 0.0
    )
    return y_mean - slope * x_mean, slope


def forecast_weekly_sequences(
    history: pd.DataFrame,
    forecast_dates,
    target: str,
    window: int,
    decay: float,
    trend_strength: float,
    log_space: bool,
) -> np.ndarray:
    """Forecast without feeding predictions back into subsequent horizons."""
    history = history[["date", target]].sort_values("date")
    output = []
    for date in pd.DatetimeIndex(forecast_dates):
        same = history[history["date"].dt.dayofweek == date.dayofweek].tail(window)
        values = same[target].to_numpy(dtype=float)
        if len(values) < 3:
            raise ValueError(f"{target} 的星期序列历史不足")
        transformed = np.log1p(values) if log_space else values
        x = np.arange(len(values), dtype=float)
        weights = decay ** np.arange(len(values) - 1, -1, -1, dtype=float)
        intercept, slope = _weighted_line(x, transformed, weights)
        weeks_ahead = max(1.0, (date - same["date"].iloc[-1]).days / 7.0)
        prediction = intercept + slope * (
            x[-1] + trend_strength * weeks_ahead
        )
        if log_space:
            prediction = np.expm1(prediction)
        output.append(max(0.0, float(prediction)))
    return np.asarray(output)


def _daily_score(y_true, y_pred) -> float:
    error = np.abs(np.asarray(y_pred) - np.asarray(y_true)) / np.maximum(
        np.asarray(y_true, dtype=float), 1.0
    )
    return float(np.maximum(0.0, 10.0 * (1.0 - error / 0.3)).mean())


def _blend(baseline, weekly, weight: float) -> np.ndarray:
    baseline = np.maximum(np.asarray(baseline, dtype=float), 1.0)
    weekly = np.maximum(np.asarray(weekly, dtype=float), 1.0)
    return np.exp((1.0 - weight) * np.log(baseline) + weight * np.log(weekly))


def _fold_data(data: pd.DataFrame, baseline: pd.DataFrame, fold: str):
    period = pd.Period(fold, freq="M")
    history = data[data["date"] < period.start_time]
    base = baseline[
        (baseline["strategy"] == "month_end_boost")
        & (baseline["fold"].astype(str) == fold)
    ].sort_values("date")
    if base.empty:
        raise ValueError(f"缺少 {fold} 的3种子基线回测")
    return history, base


def search_target(data: pd.DataFrame, baseline: pd.DataFrame, target: str):
    rows = []
    parameter_grid = itertools.product(
        WINDOWS, DECAYS, TREND_STRENGTHS, LOG_SPACES, BLEND_WEIGHTS
    )
    for window, decay, trend, log_space, blend_weight in parameter_grid:
        fold_scores = []
        fold_deltas = []
        for fold in TUNE_FOLDS:
            history, base = _fold_data(data, baseline, fold)
            weekly = forecast_weekly_sequences(
                history, base["date"], target,
                window, decay, trend, log_space,
            )
            prediction = _blend(base[f"{target}_pred"], weekly, blend_weight)
            score = _daily_score(base[f"{target}_true"], prediction)
            base_score = _daily_score(
                base[f"{target}_true"], base[f"{target}_pred"]
            )
            fold_scores.append(score)
            fold_deltas.append(score - base_score)
        rows.append({
            "target": target,
            "window": window,
            "decay": decay,
            "trend_strength": trend,
            "log_space": log_space,
            "blend_weight": blend_weight,
            "tune_score": float(np.mean(fold_scores)),
            "tune_delta": float(np.mean(fold_deltas)),
            "winning_folds": int(sum(delta > 0 for delta in fold_deltas)),
            "worst_fold_delta": float(min(fold_deltas)),
        })

    results = pd.DataFrame(rows).sort_values(
        ["tune_score", "winning_folds", "worst_fold_delta"],
        ascending=False,
    )
    stable = results[
        (results["winning_folds"] >= 3)
        & (results["worst_fold_delta"] >= -0.10)
    ]
    selected = (stable if not stable.empty else results).iloc[0].to_dict()
    return selected, results


def predict_with_params(
    data: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    forecast_dates,
    target: str,
    params: dict,
) -> np.ndarray:
    start = pd.DatetimeIndex(forecast_dates).min()
    history = data[data["date"] < start]
    weekly = forecast_weekly_sequences(
        history,
        forecast_dates,
        target,
        int(params["window"]),
        float(params["decay"]),
        float(params["trend_strength"]),
        bool(params["log_space"]),
    )
    return _blend(
        baseline_frame[target], weekly, float(params["blend_weight"])
    )


def _round_positive(values) -> np.ndarray:
    return np.maximum(0, np.rint(values)).astype(np.int64)


def main():
    data = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    baseline_backtest = pd.read_csv(BACKTEST_PRED_PATH, parse_dates=["date"])

    selected = {}
    result_frames = []
    for target in ("purchase", "redeem"):
        params, results = search_target(data, baseline_backtest, target)
        selected[target] = params
        results.insert(0, "selected", False)
        match = (
            (results["window"] == params["window"])
            & (results["decay"] == params["decay"])
            & (results["trend_strength"] == params["trend_strength"])
            & (results["log_space"] == params["log_space"])
            & (results["blend_weight"] == params["blend_weight"])
        )
        results.loc[match, "selected"] = True
        result_frames.append(results)

    holdout_rows = []
    for target, params in selected.items():
        _, base = _fold_data(data, baseline_backtest, HOLDOUT_FOLD)
        prediction = predict_with_params(
            data, base.rename(columns={f"{target}_pred": target}),
            base["date"], target, params,
        )
        score = _daily_score(base[f"{target}_true"], prediction)
        base_score = _daily_score(base[f"{target}_true"], base[f"{target}_pred"])
        holdout_rows.append({
            "selected": True,
            "target": target,
            "window": params["window"],
            "decay": params["decay"],
            "trend_strength": params["trend_strength"],
            "log_space": params["log_space"],
            "blend_weight": params["blend_weight"],
            "tune_score": np.nan,
            "tune_delta": np.nan,
            "winning_folds": np.nan,
            "worst_fold_delta": np.nan,
            "holdout_score": score,
            "holdout_delta": score - base_score,
        })

    all_results = pd.concat(
        result_frames + [pd.DataFrame(holdout_rows)], ignore_index=True
    )
    all_results.to_csv(WEEKLY_BACKTEST_METRICS_PATH, index=False, encoding="utf-8-sig")

    baseline_submission = pd.read_csv(
        SUBMISSION_PATH,
        header=None,
        names=["report_date", "purchase", "redeem"],
    )
    forecast_dates = pd.to_datetime(
        baseline_submission["report_date"].astype(str), format="%Y%m%d"
    )
    candidate = pd.DataFrame({"report_date": baseline_submission["report_date"]})
    for target, params in selected.items():
        candidate[target] = _round_positive(predict_with_params(
            data, baseline_submission, forecast_dates, target, params
        ))
    candidate.to_csv(WEEKLY_CANDIDATE_PATH, index=False, header=False, encoding="utf-8")

    print("选择参数（只使用4-7月）：")
    for target, params in selected.items():
        print(target, {key: params[key] for key in (
            "window", "decay", "trend_strength", "log_space", "blend_weight",
            "tune_delta", "winning_folds", "worst_fold_delta",
        )})
    print("8月独立验证：")
    print(pd.DataFrame(holdout_rows)[
        ["target", "holdout_score", "holdout_delta"]
    ].to_string(index=False))
    print(f"候选文件：{WEEKLY_CANDIDATE_PATH}")


if __name__ == "__main__":
    main()
