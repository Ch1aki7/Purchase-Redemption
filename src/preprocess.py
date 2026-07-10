import pandas as pd
import numpy as np
from .config import DAILY_FEATURES_PATH
from .data_loader import load_all_tables, normalize_columns


def parse_date(series):
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofmonth"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for target in ["purchase", "redeem"]:
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f"{target}_lag_{lag}"] = df[target].shift(lag)
        for window in [3, 7, 14, 30]:
            shifted = df[target].shift(1)
            df[f"{target}_roll_mean_{window}"] = shifted.rolling(window).mean()
            df[f"{target}_roll_std_{window}"] = shifted.rolling(window).std()
    return df


def main():
    user_profile, user_balance, share_interest, bank_shibor = load_all_tables()
    user_balance = normalize_columns(user_balance)
    share_interest = normalize_columns(share_interest)
    bank_shibor = normalize_columns(bank_shibor)

    required_balance_cols = ["report_date", "total_purchase_amt", "total_redeem_amt"]
    for col in required_balance_cols:
        if col not in user_balance.columns:
            raise ValueError(f"user_balance_table 缺少必要字段：{col}")

    # 1. 聚合每日申购/赎回总额
    user_balance["date"] = parse_date(user_balance["report_date"])
    daily = (
        user_balance.groupby("date", as_index=False)
        .agg(
            purchase=("total_purchase_amt", "sum"),
            redeem=("total_redeem_amt", "sum"),
            user_count=("user_id", "nunique") if "user_id" in user_balance.columns else ("report_date", "count"),
            record_count=("report_date", "count"),
        )
        .sort_values("date")
    )

    # 2. 合并收益率
    if "mfd_date" in share_interest.columns:
        share_interest["date"] = parse_date(share_interest["mfd_date"])
        share_cols = [c for c in share_interest.columns if c != "mfd_date"]
        daily = daily.merge(share_interest[share_cols], on="date", how="left")

    # 3. 合并 Shibor
    if "mfd_date" in bank_shibor.columns:
        bank_shibor["date"] = parse_date(bank_shibor["mfd_date"])
        bank_cols = [c for c in bank_shibor.columns if c != "mfd_date"]
        daily = daily.merge(bank_shibor[bank_cols], on="date", how="left")

    # 4. 时间特征、滞后特征、滚动特征
    daily = add_time_features(daily)
    daily = add_lag_rolling_features(daily)

    # 5. 简单缺失处理
    daily = daily.sort_values("date")
    numeric_cols = daily.select_dtypes(include=[np.number]).columns
    daily[numeric_cols] = daily[numeric_cols].ffill().bfill()

    DAILY_FEATURES_PATH.parent.mkdir(exist_ok=True)
    daily.to_csv(DAILY_FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(f"已生成特征文件：{DAILY_FEATURES_PATH}")
    print(f"样本数量：{len(daily)}，字段数量：{len(daily.columns)}")


if __name__ == "__main__":
    main()
