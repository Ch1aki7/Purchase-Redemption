"""Generate leaderboard candidates around the validated redeem-only model."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import (
    DAILY_FEATURES_PATH,
    EVENT_DISTANCE_METRICS_PATH,
    MONTHLY_SHAPE_CANDIDATE_PATH,
    REDEEM_EVENT_FULL_PATH,
    REDEEM_WEIGHT_40_PATH,
    REDEEM_WEIGHT_80_PATH,
)
from .event_distance_forecast import (
    holt_winters_forecast,
    log_ensemble,
    ridge_forecast,
)


COLUMNS = ["report_date", "purchase", "redeem"]
CANDIDATES = (
    (REDEEM_EVENT_FULL_PATH, 0.60, 1.00),
    (REDEEM_WEIGHT_40_PATH, 0.40, 1.00),
    (REDEEM_WEIGHT_80_PATH, 0.80, 0.75),
)


def main():
    data = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    baseline = pd.read_csv(
        MONTHLY_SHAPE_CANDIDATE_PATH, header=None, names=COLUMNS
    )
    dates = pd.to_datetime(baseline["report_date"].astype(str), format="%Y%m%d")
    history = data[data["date"] < dates.min()]
    metrics = pd.read_csv(EVENT_DISTANCE_METRICS_PATH)
    selected = metrics[
        (metrics["row_type"] == "selected") & (metrics["target"] == "redeem")
    ]
    if len(selected) != 1:
        raise ValueError("找不到唯一的已验证赎回模型")
    params = selected.iloc[0]
    ridge_signal = ridge_forecast(
        history, dates, "redeem", json.loads(params["ridge_spec"])
    )
    hw_signal = holt_winters_forecast(
        history, dates, "redeem", json.loads(params["hw_spec"])
    )

    for path, ridge_weight, event_scale in CANDIDATES:
        candidate = baseline.copy()
        candidate["redeem"] = np.rint(log_ensemble(
            baseline["redeem"], ridge_signal, hw_signal,
            ridge_weight, 0.0, dates, event_scale,
        )).astype(np.int64)
        candidate.to_csv(path, index=False, header=False, encoding="utf-8")
        print(
            f"{path.name}: ridge_weight={ridge_weight:.2f}, "
            f"event_scale={event_scale:.2f}"
        )


if __name__ == "__main__":
    main()
