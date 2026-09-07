"""Time-ordered calibration of repeatable weekday errors in the baseline model."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .config import (
    BACKTEST_PRED_PATH,
    CALIBRATION_CANDIDATE_PATH,
    CALIBRATION_METRICS_PATH,
    SUBMISSION_PATH,
)


TUNE_FOLDS = ("2014-05", "2014-06", "2014-07")
HOLDOUT_FOLD = "2014-08"
STRENGTHS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
CAPS = (0.03, 0.05, 0.10, 0.20)
PRIOR_STRENGTHS = (3.0, 7.0, 14.0)
ESTIMATORS = ("mean", "median")
LOOKBACK_MONTHS = (1, 2, 99)


def _score(y_true, y_pred) -> float:
    error = np.abs(np.asarray(y_pred) - np.asarray(y_true)) / np.maximum(
        np.asarray(y_true, dtype=float), 1.0
    )
    return float(np.maximum(0.0, 10.0 * (1.0 - error / 0.3)).mean())


def _history_before(frame: pd.DataFrame, fold: str, lookback: int) -> pd.DataFrame:
    period = pd.Period(fold, freq="M")
    eligible = frame[frame["date"] < period.start_time].copy()
    periods = sorted(eligible["date"].dt.to_period("M").unique())
    keep = periods[-lookback:] if lookback < 99 else periods
    return eligible[eligible["date"].dt.to_period("M").isin(keep)]


def weekday_factors(
    error_history: pd.DataFrame,
    target: str,
    strength: float,
    cap: float,
    prior_strength: float,
    estimator: str,
) -> dict[int, float]:
    log_ratio = np.log(
        np.maximum(error_history[f"{target}_true"].to_numpy(float), 1.0)
        / np.maximum(error_history[f"{target}_pred"].to_numpy(float), 1.0)
    )
    work = pd.DataFrame({
        "weekday": error_history["date"].dt.dayofweek.to_numpy(),
        "log_ratio": log_ratio,
    })
    aggregate = work.groupby("weekday")["log_ratio"].agg(["mean", "median", "count"])
    global_value = float(getattr(work["log_ratio"], estimator)())
    raw = aggregate[estimator]
    shrunk = (
        aggregate["count"] * raw + prior_strength * global_value
    ) / (aggregate["count"] + prior_strength)
    lower, upper = np.log(1.0 - cap), np.log(1.0 + cap)
    corrections = np.clip(strength * shrunk, lower, upper)
    return {int(day): float(np.exp(value)) for day, value in corrections.items()}


def apply_factors(frame: pd.DataFrame, target: str, factors: dict[int, float]):
    weekday = frame["date"].dt.dayofweek
    correction = weekday.map(factors).fillna(1.0).to_numpy(float)
    return frame[f"{target}_pred"].to_numpy(float) * correction


def search_target(backtest: pd.DataFrame, target: str):
    rows = []
    for strength, cap, prior, estimator, lookback in itertools.product(
        STRENGTHS, CAPS, PRIOR_STRENGTHS, ESTIMATORS, LOOKBACK_MONTHS
    ):
        deltas = []
        scores = []
        for fold in TUNE_FOLDS:
            train = _history_before(backtest, fold, lookback)
            test = backtest[backtest["fold"].astype(str) == fold]
            factors = weekday_factors(train, target, strength, cap, prior, estimator)
            prediction = apply_factors(test, target, factors)
            score = _score(test[f"{target}_true"], prediction)
            baseline_score = _score(test[f"{target}_true"], test[f"{target}_pred"])
            scores.append(score)
            deltas.append(score - baseline_score)
        rows.append({
            "target": target,
            "strength": strength,
            "cap": cap,
            "prior_strength": prior,
            "estimator": estimator,
            "lookback_months": lookback,
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
        & (results["winning_folds"] >= 2)
        & (results["worst_fold_delta"] >= -0.05)
    ]
    # Strength zero is the explicit safe fallback.
    selected = (
        stable.iloc[0] if not stable.empty
        else results[results["strength"] == 0].iloc[0]
    ).to_dict()
    return selected, results


def _selected_mask(results: pd.DataFrame, params: dict) -> pd.Series:
    return (
        (results["strength"] == params["strength"])
        & (results["cap"] == params["cap"])
        & (results["prior_strength"] == params["prior_strength"])
        & (results["estimator"] == params["estimator"])
        & (results["lookback_months"] == params["lookback_months"])
    )


def main():
    predictions = pd.read_csv(BACKTEST_PRED_PATH, parse_dates=["date"])
    baseline = predictions[predictions["strategy"] == "month_end_boost"].copy()
    selected = {}
    result_frames = []
    holdout_rows = []

    for target in ("purchase", "redeem"):
        params, results = search_target(baseline, target)
        results.insert(0, "selected", False)
        results.loc[_selected_mask(results, params), "selected"] = True
        selected[target] = params
        result_frames.append(results)

        train = _history_before(baseline, HOLDOUT_FOLD, int(params["lookback_months"]))
        test = baseline[baseline["fold"].astype(str) == HOLDOUT_FOLD]
        factors = weekday_factors(
            train, target,
            float(params["strength"]), float(params["cap"]),
            float(params["prior_strength"]), str(params["estimator"]),
        )
        calibrated = apply_factors(test, target, factors)
        holdout_delta = (
            _score(test[f"{target}_true"], calibrated)
            - _score(test[f"{target}_true"], test[f"{target}_pred"])
        )
        holdout_rows.append({
            "selected": True,
            **{key: params[key] for key in (
                "target", "strength", "cap", "prior_strength", "estimator",
                "lookback_months", "tune_score", "tune_delta",
                "winning_folds", "worst_fold_delta",
            )},
            "holdout_delta": holdout_delta,
            "accepted": bool(params["strength"] > 0 and holdout_delta > 0),
        })

    pd.concat(result_frames + [pd.DataFrame(holdout_rows)], ignore_index=True).to_csv(
        CALIBRATION_METRICS_PATH, index=False, encoding="utf-8-sig"
    )

    submission = pd.read_csv(
        SUBMISSION_PATH, header=None,
        names=["report_date", "purchase", "redeem"],
    )
    future = pd.DataFrame({
        "date": pd.to_datetime(submission["report_date"].astype(str), format="%Y%m%d")
    })
    candidate = submission.copy()
    for row in holdout_rows:
        target = row["target"]
        if not row["accepted"]:
            continue
        params = selected[target]
        train = baseline
        if int(params["lookback_months"]) < 99:
            periods = sorted(train["date"].dt.to_period("M").unique())
            train = train[train["date"].dt.to_period("M").isin(
                periods[-int(params["lookback_months"]):]
            )]
        factors = weekday_factors(
            train, target,
            float(params["strength"]), float(params["cap"]),
            float(params["prior_strength"]), str(params["estimator"]),
        )
        correction = future["date"].dt.dayofweek.map(factors).fillna(1.0)
        candidate[target] = np.maximum(
            0, np.rint(submission[target].to_numpy(float) * correction)
        ).astype(np.int64)
    candidate.to_csv(
        CALIBRATION_CANDIDATE_PATH, index=False, header=False, encoding="utf-8"
    )

    print("按4-7月顺序验证选择的参数，以及8月独立门禁：")
    print(pd.DataFrame(holdout_rows)[[
        "target", "strength", "cap", "prior_strength", "estimator",
        "lookback_months", "tune_delta", "winning_folds",
        "worst_fold_delta", "holdout_delta", "accepted",
    ]].to_string(index=False))
    print(f"候选文件：{CALIBRATION_CANDIDATE_PATH}")


if __name__ == "__main__":
    main()
