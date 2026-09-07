"""Transfer holiday-position residuals across comparable three-day holidays."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_forecast import _fold, _forecast_from_params, _score
from .config import (
    BACKTEST_PRED_PATH,
    CALENDAR_METRICS_PATH,
    DAILY_FEATURES_PATH,
    EVENT_ADJUST_CANDIDATE_PATH,
    EVENT_ADJUST_METRICS_PATH,
    MONTHLY_SHAPE_CANDIDATE_PATH,
)


EVENTS = {
    "qingming": {
        "pre1": "2014-04-04", "holiday1": "2014-04-05",
        "holiday2": "2014-04-06", "holiday3": "2014-04-07",
        "post1": "2014-04-08",
    },
    "labor": {
        "pre1": "2014-04-30", "holiday1": "2014-05-01",
        "holiday2": "2014-05-02", "holiday3": "2014-05-03",
        "post1": "2014-05-04",
    },
    "dragon_boat": {
        "pre1": "2014-05-30", "holiday1": "2014-05-31",
        "holiday2": "2014-06-01", "holiday3": "2014-06-02",
        "post1": "2014-06-03",
    },
}
MID_AUTUMN = {
    "pre1": "2014-09-05", "holiday1": "2014-09-06",
    "holiday2": "2014-09-07", "holiday3": "2014-09-08",
    "post1": "2014-09-09",
}
STRENGTHS = (0.25, 0.50, 0.75, 1.0)


def frozen_reference_params() -> dict[str, dict]:
    metrics = pd.read_csv(CALENDAR_METRICS_PATH)
    selected = metrics[metrics["accepted"].notna()]
    if len(selected) != 2 or set(selected["target"]) != {"purchase", "redeem"}:
        raise ValueError("日历基线指标中没有唯一的申购、赎回冻结参数")
    return {row["target"]: row.to_dict() for _, row in selected.iterrows()}


def historical_reference(data, baseline, target: str, params: dict) -> pd.DataFrame:
    frames = []
    for fold in ("2014-04", "2014-05", "2014-06"):
        _, test = _fold(data, baseline, fold)
        reference = _forecast_from_params(
            data, test["date"], test[f"{target}_pred"], target, params
        )
        frames.append(pd.DataFrame({
            "date": test["date"],
            "actual": test[f"{target}_true"].to_numpy(float),
            "reference": reference,
        }))
    return pd.concat(frames, ignore_index=True).set_index("date")


def event_residuals(reference: pd.DataFrame) -> dict[str, dict[str, float]]:
    residuals = {}
    for event, roles in EVENTS.items():
        residuals[event] = {
            role: float(
                reference.at[pd.Timestamp(date), "actual"]
                / reference.at[pd.Timestamp(date), "reference"]
            )
            for role, date in roles.items()
        }
    return residuals


def correction_from_events(residuals, event_names, role: str, strength: float):
    log_ratios = [np.log(residuals[event][role]) for event in event_names]
    return float(np.exp(strength * np.median(log_ratios)))


def validate_strength(reference, residuals, strength: float):
    deltas = []
    event_deltas = {}
    names = list(EVENTS)
    for held_out in names:
        train_events = [name for name in names if name != held_out]
        dates = [pd.Timestamp(date) for date in EVENTS[held_out].values()]
        roles = list(EVENTS[held_out])
        actual = reference.loc[dates, "actual"].to_numpy(float)
        baseline = reference.loc[dates, "reference"].to_numpy(float)
        adjusted = baseline * np.asarray([
            correction_from_events(residuals, train_events, role, strength)
            for role in roles
        ])
        delta = _score(actual, adjusted) - _score(actual, baseline)
        event_deltas[held_out] = float(delta)
        deltas.append(delta)
    return {
        "strength": strength,
        "loo_delta": float(np.mean(deltas)),
        "winning_events": int(sum(delta > 0 for delta in deltas)),
        "worst_event_delta": float(min(deltas)),
        **{f"{event}_delta": delta for event, delta in event_deltas.items()},
    }


def main():
    data = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    predictions = pd.read_csv(BACKTEST_PRED_PATH, parse_dates=["date"])
    baseline = predictions[predictions["strategy"] == "month_end_boost"].copy()
    params = frozen_reference_params()
    selections = {}
    metric_rows = []

    for target in ("purchase", "redeem"):
        reference = historical_reference(data, baseline, target, params[target])
        residuals = event_residuals(reference)
        results = pd.DataFrame([
            {"row_type": "grid", "target": target, **validate_strength(
                reference, residuals, strength
            )}
            for strength in STRENGTHS
        ])
        stable = results[
            (results["loo_delta"] > 0)
            & (results["winning_events"] >= 2)
            & (results["worst_event_delta"] >= -0.50)
        ]
        if stable.empty:
            choice = {"target": target, "strength": 0.0, "accepted": False}
        else:
            choice = stable.sort_values("loo_delta", ascending=False).iloc[0].to_dict()
            choice["accepted"] = True
        choice["row_type"] = "selected"
        selections[target] = (choice, residuals)
        metric_rows.extend(results.to_dict("records"))
        metric_rows.append(choice)

    pd.DataFrame(metric_rows).to_csv(
        EVENT_ADJUST_METRICS_PATH, index=False, encoding="utf-8-sig"
    )
    candidate = pd.read_csv(
        MONTHLY_SHAPE_CANDIDATE_PATH,
        header=None,
        names=["report_date", "purchase", "redeem"],
    )
    date_index = pd.to_datetime(candidate["report_date"].astype(str), format="%Y%m%d")
    candidate.index = date_index
    all_events = list(EVENTS)
    for target, (choice, residuals) in selections.items():
        if not choice["accepted"]:
            continue
        for role, date in MID_AUTUMN.items():
            correction = correction_from_events(
                residuals, all_events, role, float(choice["strength"])
            )
            candidate.loc[pd.Timestamp(date), target] = int(np.rint(
                candidate.loc[pd.Timestamp(date), target] * correction
            ))
    candidate.reset_index(drop=True).to_csv(
        EVENT_ADJUST_CANDIDATE_PATH, index=False, header=False, encoding="utf-8"
    )

    selected_rows = pd.DataFrame([value[0] for value in selections.values()])
    print(selected_rows.reindex(columns=[
        "target", "strength", "loo_delta", "winning_events",
        "worst_event_delta", "accepted",
    ]).to_string(index=False))
    print(f"中秋事件候选：{EVENT_ADJUST_CANDIDATE_PATH}")


if __name__ == "__main__":
    main()
