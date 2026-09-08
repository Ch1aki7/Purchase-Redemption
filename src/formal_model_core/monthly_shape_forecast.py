"""Forecast monthly volume and normalized daily shape as separate components."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .calendar_forecast import _fold, _forecast_from_params, _score
from .config import (
    BACKTEST_PRED_PATH,
    CALENDAR_CANDIDATE_PATH,
    CALENDAR_METRICS_PATH,
    DAILY_FEATURES_PATH,
    MONTHLY_SHAPE_CANDIDATE_PATH,
    MONTHLY_SHAPE_METRICS_PATH,
)
from .predict import HOLIDAYS_2014


DEVELOPMENT_FOLDS = ("2014-04", "2014-05", "2014-06", "2014-07")
HOLDOUT_FOLD = "2014-08"
TOTAL_WEIGHTS = (0.0, 0.10, 0.25, 0.40, 0.60, 0.80, 1.0)
SHAPE_WEIGHTS = (0.0, 0.10, 0.25, 0.40, 0.60)
TOP_MODELS = 6


def frozen_reference_params() -> dict[str, dict]:
    metrics = pd.read_csv(CALENDAR_METRICS_PATH)
    selected = metrics[metrics["accepted"].notna()]
    if len(selected) != 2 or set(selected["target"]) != {"purchase", "redeem"}:
        raise ValueError("日历基线指标中没有唯一的申购、赎回冻结参数")
    return {row["target"]: row.to_dict() for _, row in selected.iterrows()}


def total_specs() -> list[dict]:
    specs = []
    for method in ("mean", "median"):
        for months in (2, 3, 4, 6, 9, 12):
            specs.append({"method": method, "months": months, "alpha": 0.0})
    for months, alpha in itertools.product((4, 6, 9, 12), (0.1, 1.0, 10.0)):
        specs.append({"method": "log_trend", "months": months, "alpha": alpha})
    return specs


def shape_specs() -> list[dict]:
    return [
        {"months": months, "alpha": alpha, "log_space": log_space}
        for months, alpha, log_space in itertools.product(
            (3, 4, 6, 9, 12), (0.1, 1.0, 10.0, 100.0), (False, True)
        )
    ]


def monthly_levels(history: pd.DataFrame, target: str) -> pd.DataFrame:
    frame = history[["date", target]].dropna().copy()
    frame["period"] = frame["date"].dt.to_period("M")
    monthly = frame.groupby("period")[target].agg(["sum", "count"]).reset_index()
    monthly["daily_level"] = monthly["sum"] / monthly["count"]
    # The first source month is incomplete in some installations and must not
    # be treated as a normal monthly level.
    expected = monthly["period"].dt.days_in_month
    return monthly[monthly["count"] == expected].reset_index(drop=True)


def forecast_total(history, dates, target: str, spec: dict) -> float:
    monthly = monthly_levels(history, target).tail(int(spec["months"]))
    if len(monthly) < 2:
        raise ValueError("可用于月度总量预测的完整历史月份不足")
    values = monthly["daily_level"].to_numpy(dtype=float)
    if spec["method"] == "mean":
        level = float(values.mean())
    elif spec["method"] == "median":
        level = float(np.median(values))
    else:
        x = np.arange(len(values), dtype=float).reshape(-1, 1)
        model = Ridge(alpha=float(spec["alpha"]))
        model.fit(x, np.log1p(values))
        level = float(np.expm1(model.predict([[len(values)]])[0]))
    return max(level, 1.0) * len(pd.DatetimeIndex(dates))


def shape_matrix(dates) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    weekday = dates.dayofweek.to_numpy()
    day = dates.day.to_numpy()
    days_in_month = dates.days_in_month.to_numpy()
    progress = (day - 1) / np.maximum(days_in_month - 1, 1)
    columns = [
        *[(weekday == value).astype(float) for value in range(7)],
        *[(day == value).astype(float) for value in range(1, 32)],
        *[((day - 1) // 7 == value).astype(float) for value in range(5)],
        np.sin(2 * np.pi * progress),
        np.cos(2 * np.pi * progress),
        np.sin(4 * np.pi * progress),
        np.cos(4 * np.pi * progress),
        (day <= 3).astype(float),
        (day >= days_in_month - 2).astype(float),
        np.asarray(
            [date.strftime("%Y-%m-%d") in HOLIDAYS_2014 for date in dates],
            dtype=float,
        ),
    ]
    return np.column_stack(columns)


def forecast_shape(history, dates, target: str, spec: dict) -> np.ndarray:
    monthly = monthly_levels(history, target)
    keep_periods = set(monthly.tail(int(spec["months"]))["period"])
    train = history[history["date"].dt.to_period("M").isin(keep_periods)].copy()
    period = train["date"].dt.to_period("M")
    means = train.groupby(period)[target].transform("mean")
    relative = np.maximum(train[target].to_numpy(float) / means.to_numpy(float), 1e-6)
    y = np.log(relative) if spec["log_space"] else relative
    model = Ridge(alpha=float(spec["alpha"]))
    model.fit(shape_matrix(train["date"]), y)
    prediction = model.predict(shape_matrix(dates))
    if spec["log_space"]:
        prediction = np.exp(prediction)
    prediction = np.maximum(prediction, 1e-6)
    return prediction / prediction.sum()


def geometric(values_a, values_b, weight: float):
    a = np.maximum(np.asarray(values_a, dtype=float), 1e-12)
    b = np.maximum(np.asarray(values_b, dtype=float), 1e-12)
    return np.exp((1.0 - weight) * np.log(a) + weight * np.log(b))


def structural_prediction(
    reference, model_total: float, model_shape, total_weight: float, shape_weight: float
):
    reference = np.maximum(np.asarray(reference, dtype=float), 1.0)
    reference_total = float(reference.sum())
    reference_shape = reference / reference_total
    total = float(geometric(reference_total, model_total, total_weight))
    shape = geometric(reference_shape, model_shape, shape_weight)
    shape /= shape.sum()
    return total * shape


def reference_folds(data, baseline, target, params):
    result = {}
    for fold in (*DEVELOPMENT_FOLDS, HOLDOUT_FOLD):
        history, test = _fold(data, baseline, fold)
        reference = _forecast_from_params(
            data, test["date"], test[f"{target}_pred"], target, params
        )
        result[fold] = (history, test, reference)
    return result


def encode_spec(spec: dict) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True)


def shortlist_components(folds, target: str):
    total_rows = []
    for spec in total_specs():
        errors = []
        for fold in DEVELOPMENT_FOLDS:
            history, test, _ = folds[fold]
            prediction = forecast_total(history, test["date"], target, spec)
            actual = float(test[f"{target}_true"].sum())
            errors.append(abs(prediction - actual) / actual)
        total_rows.append({"spec": encode_spec(spec), "error": np.mean(errors)})

    shape_rows = []
    for spec in shape_specs():
        deltas = []
        for fold in DEVELOPMENT_FOLDS:
            history, test, reference = folds[fold]
            actual = test[f"{target}_true"].to_numpy(float)
            actual_total = actual.sum()
            model_prediction = actual_total * forecast_shape(
                history, test["date"], target, spec
            )
            reference_prediction = actual_total * reference / reference.sum()
            deltas.append(_score(actual, model_prediction) - _score(actual, reference_prediction))
        shape_rows.append({"spec": encode_spec(spec), "delta": np.mean(deltas)})

    totals = pd.DataFrame(total_rows).nsmallest(TOP_MODELS, "error")
    shapes = pd.DataFrame(shape_rows).nlargest(TOP_MODELS, "delta")
    return totals, shapes


def select_target(data, baseline, target: str, params: dict):
    folds = reference_folds(data, baseline, target, params)
    totals, shapes = shortlist_components(folds, target)
    cache = {}
    for fold in DEVELOPMENT_FOLDS:
        history, test, _ = folds[fold]
        for encoded in totals["spec"]:
            cache[(fold, "total", encoded)] = forecast_total(
                history, test["date"], target, json.loads(encoded)
            )
        for encoded in shapes["spec"]:
            cache[(fold, "shape", encoded)] = forecast_shape(
                history, test["date"], target, json.loads(encoded)
            )

    rows = []
    for total_spec, shape_spec, total_weight, shape_weight in itertools.product(
        totals["spec"], shapes["spec"], TOTAL_WEIGHTS, SHAPE_WEIGHTS
    ):
        if total_weight == 0 and shape_weight == 0:
            continue
        deltas = []
        for fold in DEVELOPMENT_FOLDS:
            _, test, reference = folds[fold]
            prediction = structural_prediction(
                reference,
                cache[(fold, "total", total_spec)],
                cache[(fold, "shape", shape_spec)],
                total_weight,
                shape_weight,
            )
            deltas.append(
                _score(test[f"{target}_true"], prediction)
                - _score(test[f"{target}_true"], reference)
            )
        rows.append({
            "target": target,
            "total_spec": total_spec,
            "shape_spec": shape_spec,
            "total_weight": total_weight,
            "shape_weight": shape_weight,
            "development_delta": float(np.mean(deltas)),
            "winning_folds": int(sum(delta > 0 for delta in deltas)),
            "worst_fold_delta": float(min(deltas)),
            **{
                f"{fold}_delta": float(delta)
                for fold, delta in zip(DEVELOPMENT_FOLDS, deltas)
            },
        })
    results = pd.DataFrame(rows)
    safe = results[
        (results["development_delta"] > 0)
        & (results["winning_folds"] >= 3)
        & (results["worst_fold_delta"] >= -0.05)
    ]
    if safe.empty:
        return None, results, folds
    return safe.sort_values(
        ["development_delta", "winning_folds", "worst_fold_delta"], ascending=False
    ).iloc[0].to_dict(), results, folds


def predict_selected(history, dates, target, reference, selected):
    total_spec = json.loads(selected["total_spec"])
    shape_spec = json.loads(selected["shape_spec"])
    return structural_prediction(
        reference,
        forecast_total(history, dates, target, total_spec),
        forecast_shape(history, dates, target, shape_spec),
        float(selected["total_weight"]),
        float(selected["shape_weight"]),
    )


def main():
    data = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    backtest = pd.read_csv(BACKTEST_PRED_PATH, parse_dates=["date"])
    baseline = backtest[backtest["strategy"] == "month_end_boost"].copy()
    reference_params = frozen_reference_params()
    selections = {}
    frames = []

    for target in ("purchase", "redeem"):
        selected, results, folds = select_target(
            data, baseline, target, reference_params[target]
        )
        results.insert(0, "row_type", "grid")
        frames.append(results)
        if selected is None:
            selections[target] = {"target": target, "accepted": False}
            continue
        history, test, reference = folds[HOLDOUT_FOLD]
        prediction = predict_selected(
            history, test["date"], target, reference, selected
        )
        selected["holdout_delta"] = (
            _score(test[f"{target}_true"], prediction)
            - _score(test[f"{target}_true"], reference)
        )
        selected["accepted"] = bool(selected["holdout_delta"] > 0)
        selections[target] = selected

    summary = pd.DataFrame(selections.values())
    summary.insert(0, "row_type", "selected")
    pd.concat([*frames, summary], ignore_index=True).to_csv(
        MONTHLY_SHAPE_METRICS_PATH, index=False, encoding="utf-8-sig"
    )

    reference = pd.read_csv(
        CALENDAR_CANDIDATE_PATH,
        header=None,
        names=["report_date", "purchase", "redeem"],
    )
    dates = pd.to_datetime(reference["report_date"].astype(str), format="%Y%m%d")
    history = data[data["date"] < dates.min()]
    candidate = reference.copy()
    for target, selected in selections.items():
        if not selected.get("accepted", False):
            continue
        candidate[target] = np.rint(predict_selected(
            history, dates, target, reference[target], selected
        )).astype(np.int64)
    candidate.to_csv(
        MONTHLY_SHAPE_CANDIDATE_PATH, index=False, header=False, encoding="utf-8"
    )

    display = [
        "target", "total_weight", "shape_weight", "development_delta",
        "winning_folds", "worst_fold_delta", "holdout_delta", "accepted",
    ]
    print(summary.reindex(columns=["row_type", *display]).to_string(index=False))
    print(f"候选文件：{MONTHLY_SHAPE_CANDIDATE_PATH}")


if __name__ == "__main__":
    main()
