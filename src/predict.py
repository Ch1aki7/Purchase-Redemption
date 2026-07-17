import json
import joblib
import numpy as np
import pandas as pd
from .config import (
    DAILY_FEATURES_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_PURCHASE_PATH,
    MODEL_REDEEM_PATH,
    SUBMISSION_PATH,
    SUBMISSION_WITH_HEADER_PATH,
)


def date_features(date):
    return {
        "dayofweek": date.dayofweek,
        "dayofmonth": date.day,
        "month": date.month,
        "is_weekend": int(date.dayofweek in [5, 6]),
        "is_month_start": int(date.is_month_start),
        "is_month_end": int(date.is_month_end),
    }


def build_future_row(date, history, feature_cols, exog_values):
    row = {c: 0 for c in feature_cols}
    row.update(date_features(date))
    row.update(exog_values)

    for target in ["purchase", "redeem"]:
        values = history[target].astype(float).tolist()
        for lag in [1, 2, 3, 7, 14, 30]:
            row[f"{target}_lag_{lag}"] = values[-lag] if len(values) >= lag else values[-1]
        for window in [3, 7, 14, 30]:
            recent = values[-window:] if len(values) >= window else values
            row[f"{target}_roll_mean_{window}"] = float(np.mean(recent))
            row[f"{target}_roll_std_{window}"] = float(np.std(recent))
    return pd.DataFrame([row])[feature_cols].fillna(0)


def main():
    for path in [DAILY_FEATURES_PATH, FEATURE_COLUMNS_PATH, MODEL_PURCHASE_PATH, MODEL_REDEEM_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"缺少必要文件：{path}。请先运行 python run_all.py 中的预处理和训练步骤。")

    df = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values("date")

    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    model_purchase = joblib.load(MODEL_PURCHASE_PATH)
    model_redeem = joblib.load(MODEL_REDEEM_PATH)

    history = df[["date", "purchase", "redeem"]].copy()

    # 9 月未来收益率、Shibor 不一定在原始数据里，这里使用最后一天的外部变量做前向填充。
    non_exog = {"date", "purchase", "redeem"}
    lag_roll_prefixes = ("purchase_lag_", "redeem_lag_", "purchase_roll_", "redeem_roll_")
    time_cols = {"dayofweek", "dayofmonth", "month", "is_weekend", "is_month_start", "is_month_end"}
    exog_cols = [
        c for c in feature_cols
        if c not in non_exog and c not in time_cols and not c.startswith(lag_roll_prefixes)
    ]
    last_row = df.iloc[-1]
    exog_values = {c: last_row[c] for c in exog_cols if c in df.columns}

    pred_rows = []
    for date in pd.date_range("2014-09-01", "2014-09-30", freq="D"):
        X = build_future_row(date, history, feature_cols, exog_values)
        purchase_pred = float(np.expm1(model_purchase.predict(X)[0]))
        redeem_pred = float(np.expm1(model_redeem.predict(X)[0]))
        purchase_pred = max(0, purchase_pred)
        redeem_pred = max(0, redeem_pred)

        pred_rows.append({
            "report_date": int(date.strftime("%Y%m%d")),
            "purchase": int(round(purchase_pred)),
            "redeem": int(round(redeem_pred)),
        })
        history = pd.concat([
            history,
            pd.DataFrame({"date": [date], "purchase": [purchase_pred], "redeem": [redeem_pred]})
        ], ignore_index=True)

    submission = pd.DataFrame(pred_rows)
    # 天池提交样例通常不带表头，因此 tc_comp_predict_table.csv 默认保存为无表头版本。
    submission.to_csv(SUBMISSION_PATH, index=False, header=False, encoding="utf-8-sig")
    # 额外保存一个带表头版本，便于在报告和可视化页面中查看。
    submission.to_csv(SUBMISSION_WITH_HEADER_PATH, index=False, encoding="utf-8-sig")
    print(f"预测提交文件已保存到：{SUBMISSION_PATH}")
    print(f"带表头查看文件已保存到：{SUBMISSION_WITH_HEADER_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
