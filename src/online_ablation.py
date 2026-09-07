"""Create target-isolated submissions for interpreting leaderboard feedback."""

from __future__ import annotations

import pandas as pd

from .config import (
    EVENT_DISTANCE_CANDIDATE_PATH,
    EVENT_DISTANCE_PURCHASE_ONLY_PATH,
    EVENT_DISTANCE_REDEEM_ONLY_PATH,
    MONTHLY_SHAPE_CANDIDATE_PATH,
)


COLUMNS = ["report_date", "purchase", "redeem"]


def load_submission(path):
    frame = pd.read_csv(path, header=None, names=COLUMNS)
    if frame.shape != (30, 3):
        raise ValueError(f"{path} 不是30行3列")
    if frame["report_date"].tolist() != list(range(20140901, 20140931)):
        raise ValueError(f"{path} 日期不完整或顺序错误")
    if not (frame[["purchase", "redeem"]] > 0).all().all():
        raise ValueError(f"{path} 存在非正预测值")
    return frame


def main():
    baseline = load_submission(MONTHLY_SHAPE_CANDIDATE_PATH)
    full = load_submission(EVENT_DISTANCE_CANDIDATE_PATH)
    if not baseline["report_date"].equals(full["report_date"]):
        raise ValueError("基线和新模型日期不一致")

    purchase_only = baseline.copy()
    purchase_only["purchase"] = full["purchase"]
    purchase_only.to_csv(
        EVENT_DISTANCE_PURCHASE_ONLY_PATH,
        index=False, header=False, encoding="utf-8",
    )

    redeem_only = baseline.copy()
    redeem_only["redeem"] = full["redeem"]
    redeem_only.to_csv(
        EVENT_DISTANCE_REDEEM_ONLY_PATH,
        index=False, header=False, encoding="utf-8",
    )
    print(f"仅替换申购：{EVENT_DISTANCE_PURCHASE_ONLY_PATH}")
    print(f"仅替换赎回：{EVENT_DISTANCE_REDEEM_ONLY_PATH}")


if __name__ == "__main__":
    main()
