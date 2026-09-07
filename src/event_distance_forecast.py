"""Event-distance regression and weekly Holt-Winters ensemble.

Model families are ranked on overlapping 30-day rolling-origin folds.  The
final ensemble is chosen on April-July monthly folds; August is diagnostic only
because it is known to be an atypical validation month for this competition.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .calendar_forecast import _fold, _forecast_from_params, _score
from .config import (
    BACKTEST_PRED_PATH,
    CALENDAR_METRICS_PATH,
    DAILY_FEATURES_PATH,
    EVENT_DISTANCE_CANDIDATE_PATH,
    EVENT_DISTANCE_METRICS_PATH,
    MONTHLY_SHAPE_CANDIDATE_PATH,
)


ROLLING_STARTS = (
    "2014-04-01", "2014-04-15", "2014-05-01", "2014-05-15",
    "2014-06-01", "2014-06-15", "2014-07-01", "2014-07-15",
    "2014-08-01",
)
TUNE_MONTHS = ("2014-04", "2014-05", "2014-06", "2014-07")
HOLDOUT_MONTH = "2014-08"
TOP_SIGNALS = 4
EVENT_SCALES = (0.0, 0.25, 0.50, 0.75, 1.0)

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


def event_matrix(dates) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    starts = pd.DatetimeIndex([start for start, _ in HOLIDAY_PERIODS])
    ends = pd.DatetimeIndex([end for _, end in HOLIDAY_PERIODS])
    special_work = pd.DatetimeIndex(sorted(SPECIAL_WORKDAYS))
    weekday = dates.dayofweek.to_numpy()
    day = dates.day.to_numpy()
    dim = dates.days_in_month.to_numpy()
    trend = (dates - pd.Timestamp("2013-07-01")).days.to_numpy(float) / 365.0
    is_holiday = np.zeros(len(dates), dtype=float)
    position = np.zeros(len(dates), dtype=int)
    holiday_length = np.zeros(len(dates), dtype=int)
    for start_text, end_text in HOLIDAY_PERIODS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        mask = (dates >= start) & (dates <= end)
        is_holiday[mask] = 1.0
        position[mask] = (dates[mask] - start).days + 1
        holiday_length[mask] = (end - start).days + 1
    is_special_work = dates.isin(special_work).astype(float)
    is_work = (((weekday < 5) & (is_holiday == 0)) | (is_special_work == 1)).astype(float)

    # Ignore periods on the wrong side of the date when computing distances.
    to_start = np.asarray([
        min([(start - date).days for start in starts if start >= date] or [30])
        for date in dates
    ])
    from_end = np.asarray([
        min([(date - end).days for end in ends if end <= date] or [30])
        for date in dates
    ])
    progress = (day - 1) / np.maximum(dim - 1, 1)
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
        (day <= 3).astype(float), (day >= dim - 2).astype(float),
        *[(day == value).astype(float) for value in (5, 10, 15, 20, 25)],
    ]
    columns.extend([
        (weekday == value).astype(float) * is_work for value in range(7)
    ])
    return np.column_stack(columns)


def ridge_specs():
    for window, log_space, alpha, half_life in itertools.product(
        (120, 180, 240, 365, 427), (False, True),
        (0.1, 1.0, 10.0, 100.0), (60.0, 120.0, 240.0, 10000.0),
    ):
        yield {
            "window": window, "log_space": log_space,
            "alpha": alpha, "half_life": half_life,
        }


def ridge_forecast(history, dates, target: str, spec: dict):
    train = history.sort_values("date").tail(int(spec["window"]))
    y = train[target].to_numpy(float)
    if spec["log_space"]:
        y = np.log1p(y)
    age = (train["date"].max() - train["date"]).dt.days.to_numpy(float)
    weights = np.exp(-np.log(2.0) * age / float(spec["half_life"]))
    model = Ridge(alpha=float(spec["alpha"]))
    model.fit(event_matrix(train["date"]), y, sample_weight=weights)
    prediction = model.predict(event_matrix(dates))
    if spec["log_space"]:
        prediction = np.expm1(prediction)
    bounds = np.percentile(train[target], [1, 99])
    return np.clip(prediction, max(1.0, bounds[0] * 0.5), bounds[1] * 1.5)


def hw_specs():
    for window, alpha, beta, gamma, damping, log_space in itertools.product(
        (84, 140, 224, 365), (0.1, 0.25, 0.5), (0.0, 0.05),
        (0.1, 0.3, 0.5), (0.95, 1.0), (False, True),
    ):
        yield {
            "window": window, "alpha": alpha, "beta": beta,
            "gamma": gamma, "damping": damping, "log_space": log_space,
        }


def holt_winters_forecast(history, dates, target: str, spec: dict):
    raw = history.sort_values("date")[target].tail(int(spec["window"])).to_numpy(float)
    values = np.log1p(raw) if spec["log_space"] else raw.copy()
    season = 7
    if len(values) < season * 2:
        raise ValueError("Holt-Winters 历史不足")
    level = float(np.mean(values[:season]))
    trend = float((np.mean(values[season:2 * season]) - level) / season)
    seasonal = np.asarray([
        np.mean(values[pos::season] - np.mean(values)) for pos in range(season)
    ], dtype=float)
    alpha, beta = float(spec["alpha"]), float(spec["beta"])
    gamma, damping = float(spec["gamma"]), float(spec["damping"])
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
    if spec["log_space"]:
        prediction = np.expm1(prediction)
    bounds = np.percentile(raw, [1, 99])
    return np.clip(prediction, max(1.0, bounds[0] * 0.5), bounds[1] * 1.5)


def encode(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True)


def rolling_score(data, target: str, spec: dict, family: str):
    scores = []
    for start_text in ROLLING_STARTS:
        start = pd.Timestamp(start_text)
        history = data[data["date"] < start]
        actual = data[(data["date"] >= start) & (data["date"] < start + pd.Timedelta(days=30))]
        forecast = (
            ridge_forecast(history, actual["date"], target, spec)
            if family == "ridge" else
            holt_winters_forecast(history, actual["date"], target, spec)
        )
        scores.append(_score(actual[target], forecast))
    return scores


def shortlist(data, target: str, family: str):
    specs = ridge_specs() if family == "ridge" else hw_specs()
    rows = []
    for spec in specs:
        scores = rolling_score(data, target, spec, family)
        rows.append({
            "row_type": f"{family}_rolling",
            "target": target,
            "family": family,
            "spec": encode(spec),
            "rolling_score": float(np.mean(scores)),
            "rolling_median": float(np.median(scores)),
            "rolling_worst": float(np.min(scores)),
        })
    results = pd.DataFrame(rows).sort_values(
        ["rolling_score", "rolling_median", "rolling_worst"], ascending=False
    )
    return results.head(TOP_SIGNALS), results


def frozen_reference_params():
    metrics = pd.read_csv(CALENDAR_METRICS_PATH)
    selected = metrics[metrics["accepted"].notna()]
    if len(selected) != 2 or set(selected["target"]) != {"purchase", "redeem"}:
        raise ValueError("冻结日历参数不完整")
    return {row["target"]: row.to_dict() for _, row in selected.iterrows()}


def event_window_mask(dates) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    mask = dates.isin(pd.DatetimeIndex(sorted(SPECIAL_WORKDAYS)))
    for start_text, end_text in HOLIDAY_PERIODS:
        start = pd.Timestamp(start_text) - pd.Timedelta(days=1)
        end = pd.Timestamp(end_text) + pd.Timedelta(days=1)
        mask |= (dates >= start) & (dates <= end)
    return np.asarray(mask, dtype=bool)


def log_ensemble(
    reference, ridge_signal, hw_signal, ridge_weight, hw_weight,
    dates=None, event_scale: float = 1.0,
):
    reference = np.asarray(reference, dtype=float)
    ridge_signal = np.asarray(ridge_signal, dtype=float)
    hw_signal = np.asarray(hw_signal, dtype=float)
    ridge_weights = np.full(len(reference), ridge_weight, dtype=float)
    hw_weights = np.full(len(reference), hw_weight, dtype=float)
    if dates is not None and event_scale < 1.0:
        mask = event_window_mask(dates)
        ridge_weights[mask] *= event_scale
        hw_weights[mask] *= event_scale
    reference_weights = 1.0 - ridge_weights - hw_weights
    return np.exp(
        reference_weights * np.log(np.maximum(reference, 1.0))
        + ridge_weights * np.log(np.maximum(ridge_signal, 1.0))
        + hw_weights * np.log(np.maximum(hw_signal, 1.0))
    )


def select_ensemble(data, baseline, target, reference_params, ridge_top, hw_top):
    cache = {}
    for month in (*TUNE_MONTHS, HOLDOUT_MONTH):
        history, test = _fold(data, baseline, month)
        reference = _forecast_from_params(
            data, test["date"], test[f"{target}_pred"], target, reference_params
        )
        cache[(month, "test")] = test
        cache[(month, "reference")] = reference
        for spec_text in ridge_top["spec"]:
            cache[(month, "ridge", spec_text)] = ridge_forecast(
                history, test["date"], target, json.loads(spec_text)
            )
        for spec_text in hw_top["spec"]:
            cache[(month, "hw", spec_text)] = holt_winters_forecast(
                history, test["date"], target, json.loads(spec_text)
            )
    rows = []
    weights = (0.0, 0.10, 0.25, 0.40, 0.60, 0.80, 1.0)
    for ridge_spec, hw_spec, rw, hw, event_scale in itertools.product(
        ridge_top["spec"], hw_top["spec"], weights, weights, EVENT_SCALES
    ):
        if rw + hw == 0 or rw + hw > 1:
            continue
        deltas = []
        scores = []
        for month in TUNE_MONTHS:
            test = cache[(month, "test")]
            reference = cache[(month, "reference")]
            prediction = log_ensemble(
                reference, cache[(month, "ridge", ridge_spec)],
                cache[(month, "hw", hw_spec)], rw, hw,
                test["date"], event_scale,
            )
            scores.append(_score(test[f"{target}_true"], prediction))
            deltas.append(scores[-1] - _score(test[f"{target}_true"], reference))
        rows.append({
            "row_type": "ensemble_grid", "target": target,
            "ridge_spec": ridge_spec, "hw_spec": hw_spec,
            "ridge_weight": rw, "hw_weight": hw,
            "event_scale": event_scale,
            "tune_score": float(np.mean(scores)),
            "tune_delta": float(np.mean(deltas)),
            "winning_months": int(sum(delta > 0 for delta in deltas)),
            "worst_month_delta": float(min(deltas)),
            **{f"{month}_delta": delta for month, delta in zip(TUNE_MONTHS, deltas)},
        })
    grid = pd.DataFrame(rows)
    safe = grid[
        (grid["tune_delta"] > 0)
        & (grid["winning_months"] >= 3)
        & (grid["worst_month_delta"] >= -0.15)
    ]
    if safe.empty:
        return None, grid
    selected = safe.sort_values(
        ["tune_score", "winning_months", "worst_month_delta"], ascending=False
    ).iloc[0].to_dict()
    test = cache[(HOLDOUT_MONTH, "test")]
    reference = cache[(HOLDOUT_MONTH, "reference")]
    prediction = log_ensemble(
        reference,
        cache[(HOLDOUT_MONTH, "ridge", selected["ridge_spec"])],
        cache[(HOLDOUT_MONTH, "hw", selected["hw_spec"])],
        selected["ridge_weight"], selected["hw_weight"],
        test["date"], selected["event_scale"],
    )
    selected["holdout_delta"] = (
        _score(test[f"{target}_true"], prediction)
        - _score(test[f"{target}_true"], reference)
    )
    selected["row_type"] = "selected"
    return selected, grid


def main():
    data = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    backtest = pd.read_csv(BACKTEST_PRED_PATH, parse_dates=["date"])
    baseline = backtest[backtest["strategy"] == "month_end_boost"].copy()
    reference_params = frozen_reference_params()
    selections = {}
    frames = []
    for target in ("purchase", "redeem"):
        ridge_top, ridge_all = shortlist(data, target, "ridge")
        hw_top, hw_all = shortlist(data, target, "hw")
        selected, ensemble_grid = select_ensemble(
            data, baseline, target, reference_params[target], ridge_top, hw_top
        )
        frames.extend([ridge_all, hw_all, ensemble_grid])
        selections[target] = selected
    selected_frame = pd.DataFrame([
        value if value is not None else {"row_type": "selected", "target": target}
        for target, value in selections.items()
    ])
    pd.concat([*frames, selected_frame], ignore_index=True).to_csv(
        EVENT_DISTANCE_METRICS_PATH, index=False, encoding="utf-8-sig"
    )

    reference = pd.read_csv(
        MONTHLY_SHAPE_CANDIDATE_PATH, header=None,
        names=["report_date", "purchase", "redeem"],
    )
    dates = pd.to_datetime(reference["report_date"].astype(str), format="%Y%m%d")
    history = data[data["date"] < dates.min()]
    candidate = reference.copy()
    for target, selected in selections.items():
        if selected is None:
            continue
        ridge_signal = ridge_forecast(
            history, dates, target, json.loads(selected["ridge_spec"])
        )
        hw_signal = holt_winters_forecast(
            history, dates, target, json.loads(selected["hw_spec"])
        )
        candidate[target] = np.rint(log_ensemble(
            reference[target], ridge_signal, hw_signal,
            float(selected["ridge_weight"]), float(selected["hw_weight"]),
            dates, float(selected["event_scale"]),
        )).astype(np.int64)
    candidate.to_csv(
        EVENT_DISTANCE_CANDIDATE_PATH, index=False, header=False, encoding="utf-8"
    )
    print(selected_frame.reindex(columns=[
        "target", "ridge_weight", "hw_weight", "event_scale", "tune_delta",
        "winning_months", "worst_month_delta", "holdout_delta",
    ]).to_string(index=False))
    print(f"候选文件：{EVENT_DISTANCE_CANDIDATE_PATH}")


if __name__ == "__main__":
    main()
