"""Reproduce and audit the published Tianchi weekday/day cycle baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_forecast import _fold, _forecast_from_params, _score
from .config import (
    BACKTEST_PRED_PATH,
    CALENDAR_METRICS_PATH,
    DAILY_FEATURES_PATH,
    MONTHLY_SHAPE_CANDIDATE_PATH,
    PUBLIC_CYCLE_CANDIDATE_PATH,
    PUBLIC_CYCLE_METRICS_PATH,
    PUBLIC_CYCLE_RAW_PATH,
)


DEVELOPMENT_FOLDS = ("2014-04", "2014-05", "2014-06", "2014-07")
HOLDOUT_FOLD = "2014-08"
BLEND_WEIGHTS = (0.0, 0.10, 0.25, 0.40, 0.60, 0.80, 1.0)
TRAIN_START = pd.Timestamp("2014-03-01")


def frozen_reference_params() -> dict[str, dict]:
    metrics = pd.read_csv(CALENDAR_METRICS_PATH)
    selected = metrics[metrics["accepted"].notna()]
    if len(selected) != 2 or set(selected["target"]) != {"purchase", "redeem"}:
        raise ValueError("日历基线指标中没有唯一的申购、赎回冻结参数")
    return {row["target"]: row.to_dict() for _, row in selected.iterrows()}


def cycle_forecast(history, dates, target: str) -> np.ndarray:
    """Port the public generate_base rule without using future observations."""
    dates = pd.DatetimeIndex(dates)
    train = history[
        (history["date"] >= TRAIN_START) & (history["date"] < dates.min())
    ][["date", target]].dropna().copy()
    if train.empty:
        raise ValueError("周期因子没有可用历史数据")
    train["weekday"] = train["date"].dt.dayofweek
    train["day"] = train["date"].dt.day
    train["month"] = train["date"].dt.to_period("M")

    weekday_factor = train.groupby("weekday")[target].mean() / train[target].mean()
    weekday_counts = train.groupby(["day", "weekday"]).size().rename("count").reset_index()
    weekday_counts["weighted"] = (
        weekday_counts["weekday"].map(weekday_factor)
        * weekday_counts["count"]
        / train["month"].nunique()
    )
    expected_weekday_factor = weekday_counts.groupby("day")["weighted"].sum()
    day_mean = train.groupby("day")[target].mean()
    day_base = day_mean / expected_weekday_factor

    fallback = float(train[target].mean())
    predictions = []
    for date in dates:
        base = float(day_base.get(date.day, fallback))
        factor = float(weekday_factor.get(date.dayofweek, 1.0))
        predictions.append(base * factor)
    return np.maximum(np.asarray(predictions, dtype=float), 1.0)


def geometric_blend(reference, cycle, weight: float):
    reference = np.maximum(np.asarray(reference, dtype=float), 1.0)
    cycle = np.maximum(np.asarray(cycle, dtype=float), 1.0)
    return np.exp((1.0 - weight) * np.log(reference) + weight * np.log(cycle))


def main():
    data = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    predictions = pd.read_csv(BACKTEST_PRED_PATH, parse_dates=["date"])
    baseline = predictions[predictions["strategy"] == "month_end_boost"].copy()
    params = frozen_reference_params()
    selected = {}
    rows = []

    for target in ("purchase", "redeem"):
        fold_cache = {}
        for fold in (*DEVELOPMENT_FOLDS, HOLDOUT_FOLD):
            history, test = _fold(data, baseline, fold)
            reference = _forecast_from_params(
                data, test["date"], test[f"{target}_pred"], target, params[target]
            )
            cycle = cycle_forecast(history, test["date"], target)
            fold_cache[fold] = (test, reference, cycle)

        candidates = []
        for weight in BLEND_WEIGHTS:
            deltas = []
            scores = []
            for fold in DEVELOPMENT_FOLDS:
                test, reference, cycle = fold_cache[fold]
                blended = geometric_blend(reference, cycle, weight)
                scores.append(_score(test[f"{target}_true"], blended))
                deltas.append(
                    scores[-1] - _score(test[f"{target}_true"], reference)
                )
            candidates.append({
                "row_type": "grid",
                "target": target,
                "weight": weight,
                "development_score": float(np.mean(scores)),
                "development_delta": float(np.mean(deltas)),
                "winning_folds": int(sum(delta > 0 for delta in deltas)),
                "worst_fold_delta": float(min(deltas)),
                **{
                    f"{fold}_delta": float(delta)
                    for fold, delta in zip(DEVELOPMENT_FOLDS, deltas)
                },
            })
        grid = pd.DataFrame(candidates)
        stable = grid[
            (grid["development_delta"] > 0)
            & (grid["winning_folds"] >= 3)
            & (grid["worst_fold_delta"] >= -0.15)
        ]
        choice = (
            stable.sort_values("development_score", ascending=False).iloc[0]
            if not stable.empty else grid[grid["weight"] == 0].iloc[0]
        ).to_dict()
        test, reference, cycle = fold_cache[HOLDOUT_FOLD]
        raw_score = _score(test[f"{target}_true"], cycle)
        prediction = geometric_blend(reference, cycle, choice["weight"])
        choice.update({
            "row_type": "selected",
            "holdout_score": _score(test[f"{target}_true"], prediction),
            "holdout_delta": (
                _score(test[f"{target}_true"], prediction)
                - _score(test[f"{target}_true"], reference)
            ),
            "raw_holdout_score": raw_score,
        })
        selected[target] = choice
        rows.extend(candidates)
        rows.append(choice)

    pd.DataFrame(rows).to_csv(
        PUBLIC_CYCLE_METRICS_PATH, index=False, encoding="utf-8-sig"
    )

    reference = pd.read_csv(
        MONTHLY_SHAPE_CANDIDATE_PATH,
        header=None,
        names=["report_date", "purchase", "redeem"],
    )
    dates = pd.to_datetime(reference["report_date"].astype(str), format="%Y%m%d")
    history = data[data["date"] < dates.min()]
    raw = reference.copy()
    candidate = reference.copy()
    for target in ("purchase", "redeem"):
        cycle = cycle_forecast(history, dates, target)
        raw[target] = np.rint(cycle).astype(np.int64)
        candidate[target] = np.rint(geometric_blend(
            reference[target], cycle, float(selected[target]["weight"])
        )).astype(np.int64)
    raw.to_csv(PUBLIC_CYCLE_RAW_PATH, index=False, header=False, encoding="utf-8")
    candidate.to_csv(
        PUBLIC_CYCLE_CANDIDATE_PATH, index=False, header=False, encoding="utf-8"
    )

    display = pd.DataFrame(selected.values())[[
        "target", "weight", "development_delta", "winning_folds",
        "worst_fold_delta", "holdout_delta", "raw_holdout_score",
    ]]
    print(display.to_string(index=False))
    print(f"公开周期原版：{PUBLIC_CYCLE_RAW_PATH}")
    print(f"与124基线融合版：{PUBLIC_CYCLE_CANDIDATE_PATH}")


if __name__ == "__main__":
    main()
