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

    # 周期连续编码（傅里叶）：让模型感知周期连续性，替代纯0/1
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["dayofmonth_sin"] = np.sin(2 * np.pi * df["dayofmonth"] / 31)
    df["dayofmonth_cos"] = np.cos(2 * np.pi * df["dayofmonth"] / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_month_start_end_features(df: pd.DataFrame) -> pd.DataFrame:
    """月初/月末连续特征：解决8月1日、8月31日误差爆炸问题。"""
    df = df.copy()
    days_in_month = df["date"].dt.days_in_month
    df["days_to_month_start"] = df["dayofmonth"] - 1
    df["days_to_month_end"] = days_in_month - df["dayofmonth"]
    df["is_first_3_days"] = (df["days_to_month_start"] < 3).astype(int)
    df["is_last_3_days"] = (df["days_to_month_end"] < 3).astype(int)
    df["is_first_7_days"] = (df["days_to_month_start"] < 7).astype(int)
    df["is_last_7_days"] = (df["days_to_month_end"] < 7).astype(int)
    # 连续权重：月初月末越近权重越大
    df["month_start_weight"] = np.exp(-df["days_to_month_start"] / 3.0)
    df["month_end_weight"] = np.exp(-df["days_to_month_end"] / 3.0)
    # 月初/月末的历史均值特征（shift避免数据泄露）
    for target in ["purchase", "redeem"]:
        # 过去3个月月初3天的均值
        df[f"{target}_month_start_mean"] = df[target].shift(1).rolling(30).mean()
    return df


# 2014年中国法定节假日 + 重要调休日
HOLIDAYS_2014 = {
    "2014-01-01", "2014-01-31", "2014-02-01", "2014-02-02", "2014-02-03",
    "2014-02-04", "2014-02-05", "2014-02-06",  # 春节
    "2014-04-05", "2014-04-06", "2014-04-07",  # 清明
    "2014-05-01", "2014-05-02", "2014-05-03",  # 劳动节
    "2014-05-31", "2014-06-01", "2014-06-02",  # 端午
    "2014-09-06", "2014-09-07", "2014-09-08",  # 中秋
    "2014-10-01", "2014-10-02", "2014-10-03", "2014-10-04",
    "2014-10-05", "2014-10-06", "2014-10-07",  # 国庆
}


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """节假日特征：2014年中秋节9月8日、国庆等。"""
    df = df.copy()
    holiday_set = pd.to_datetime(list(HOLIDAYS_2014))
    df["is_holiday"] = df["date"].isin(holiday_set).astype(int)

    # 距下一个/上一个节假日的天数
    holiday_dates = sorted(holiday_set)
    def days_to_next(d):
        future = [h for h in holiday_dates if h >= d]
        return (future[0] - d).days if future else 30
    def days_from_last(d):
        past = [h for h in holiday_dates if h <= d]
        return (d - past[-1]).days if past else 30
    df["days_to_next_holiday"] = df["date"].map(days_to_next)
    df["days_from_last_holiday"] = df["date"].map(days_from_last)
    # 节前1天标记（资金往往节前流出）
    df["is_day_before_holiday"] = (df["days_to_next_holiday"] == 1).astype(int)
    df["is_day_after_holiday"] = (df["days_from_last_holiday"] == 1).astype(int)
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
            df[f"{target}_roll_min_{window}"] = shifted.rolling(window).min()
            df[f"{target}_roll_max_{window}"] = shifted.rolling(window).max()
            df[f"{target}_roll_median_{window}"] = shifted.rolling(window).median()
    return df


def add_same_period_features(df: pd.DataFrame) -> pd.DataFrame:
    """历史同期特征：上月同日、同星期几历史均值。"""
    df = df.sort_values("date").copy()
    for target in ["purchase", "redeem"]:
        # 上月同日（lag 30 近似，但更稳定用shift）
        df[f"{target}_same_day_last_month"] = df[target].shift(30)
        # 过去4周同星期几的均值
        for week in [1, 2, 3, 4]:
            df[f"{target}_same_weekday_w{week}"] = df[target].shift(7 * week)
        df[f"{target}_mean_same_weekday_4w"] = df[
            [f"{target}_same_weekday_w{w}" for w in [1, 2, 3, 4]]
        ].mean(axis=1)
    return df


def add_ratio_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """比率与差分特征（全部用 shift 派生，避免使用当天真实值造成数据泄露）。

    注意：原版本使用当天 purchase/redeem 派生 purchase_redeem_ratio、
    purchase_diff_lag1 等特征，导致数据泄露（模型输入已包含当天答案的
    间接信息）。本版本所有特征都基于历史值（shift(1) 或更早）。
    """
    df = df.copy()
    # 昨天申购赎回比率（安全：基于昨天值）
    df["purchase_redeem_ratio_lag1"] = df["purchase"].shift(1) / (df["redeem"].shift(1) + 1)
    # 前天相对大前天的环比变化（安全：基于历史值）
    df["purchase_diff_lag1_hist"] = df["purchase"].shift(1) - df["purchase"].shift(2)
    df["redeem_diff_lag1_hist"] = df["redeem"].shift(1) - df["redeem"].shift(2)
    df["purchase_diff_lag7_hist"] = df["purchase"].shift(1) - df["purchase"].shift(8)
    df["redeem_diff_lag7_hist"] = df["redeem"].shift(1) - df["redeem"].shift(8)
    # 净流入（只用历史值，shift(1) 是昨天净流入，安全）
    df["net_inflow_lag1"] = (df["purchase"].shift(1) - df["redeem"].shift(1))
    df["net_inflow_lag7"] = (df["purchase"].shift(7) - df["redeem"].shift(7))
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

    # 4. 特征工程（顺序：时间→月初月末→节假日→滞后滚动→同期→比率差分）
    daily = add_time_features(daily)
    daily = add_month_start_end_features(daily)
    daily = add_holiday_features(daily)
    daily = add_lag_rolling_features(daily)
    daily = add_same_period_features(daily)
    daily = add_ratio_diff_features(daily)

    # 5. 简单缺失处理
    daily = daily.sort_values("date")
    numeric_cols = daily.select_dtypes(include=[np.number]).columns
    daily[numeric_cols] = daily[numeric_cols].ffill().bfill()
    # 剩余 NaN 用 0 填充（如 roll_std 在窗口不足时）
    daily[numeric_cols] = daily[numeric_cols].fillna(0)

    DAILY_FEATURES_PATH.parent.mkdir(exist_ok=True)
    daily.to_csv(DAILY_FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(f"已生成特征文件：{DAILY_FEATURES_PATH}")
    print(f"样本数量：{len(daily)}，字段数量：{len(daily.columns)}")


if __name__ == "__main__":
    main()
