import pandas as pd
import numpy as np
from .config import DAILY_FEATURES_PATH, DATA_QUALITY_REPORT_PATH
from .data_loader import load_all_tables, normalize_columns
from .calendar_features import calendar_feature_dict
from .behavior_features import add_behavior_history_features


def parse_date(series):
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")


def check_balance_consistency(user_balance: pd.DataFrame) -> pd.DataFrame:
    """校验余额一致性：tBalance = yBalance + total_purchase_amt - total_redeem_amt。

    该校验用于满足课程指导书的数据一致性要求，只输出数据质量统计，
    不会删除或修改训练样本，避免改变原始赛题数据分布。
    """
    required_cols = ["tBalance", "yBalance", "total_purchase_amt", "total_redeem_amt"]
    missing_cols = [col for col in required_cols if col not in user_balance.columns]
    if missing_cols:
        return pd.DataFrame([
            {
                "check_item": "balance_consistency",
                "status": "skipped",
                "message": f"缺少字段：{', '.join(missing_cols)}",
                "total_records": len(user_balance),
                "consistent_records": np.nan,
                "inconsistent_records": np.nan,
                "inconsistent_ratio": np.nan,
                "max_abs_error": np.nan,
                "mean_abs_error": np.nan,
            }
        ])

    balance = user_balance[required_cols].apply(pd.to_numeric, errors="coerce")
    expected = balance["yBalance"] + balance["total_purchase_amt"] - balance["total_redeem_amt"]
    diff = balance["tBalance"] - expected
    valid_mask = balance.notna().all(axis=1)
    valid_diff = diff[valid_mask]
    consistent_mask = valid_diff.abs() <= 1e-6
    inconsistent_records = int((~consistent_mask).sum())
    total_valid_records = int(valid_mask.sum())
    inconsistent_ratio = inconsistent_records / total_valid_records if total_valid_records else np.nan

    return pd.DataFrame([
        {
            "check_item": "balance_consistency",
            "status": "passed" if inconsistent_records == 0 else "warning",
            "message": "校验 tBalance = yBalance + total_purchase_amt - total_redeem_amt",
            "total_records": int(len(user_balance)),
            "valid_records": total_valid_records,
            "consistent_records": int(consistent_mask.sum()),
            "inconsistent_records": inconsistent_records,
            "inconsistent_ratio": inconsistent_ratio,
            "max_abs_error": float(valid_diff.abs().max()) if total_valid_records else np.nan,
            "mean_abs_error": float(valid_diff.abs().mean()) if total_valid_records else np.nan,
        }
    ])


def check_component_consistency(daily: pd.DataFrame) -> pd.DataFrame:
    """校验聚合后的资金流拆分恒等关系。"""
    checks = [
        ("purchase_segment_consistency", "purchase", ["large_user_purchase", "small_user_purchase"]),
        ("redeem_segment_consistency", "redeem", ["large_user_redeem", "small_user_redeem"]),
        ("redeem_component_consistency", "redeem", ["consume_amt", "transfer_amt"]),
    ]
    rows = []
    for name, total_col, component_cols in checks:
        missing = [col for col in [total_col, *component_cols] if col not in daily.columns]
        if missing:
            rows.append({
                "check_item": name,
                "status": "skipped",
                "message": f"缺少字段：{', '.join(missing)}",
                "total_records": len(daily),
            })
            continue
        diff = daily[total_col] - daily[component_cols].sum(axis=1)
        consistent = diff.abs() <= 1e-6
        rows.append({
            "check_item": name,
            "status": "passed" if bool(consistent.all()) else "warning",
            "message": f"校验 {total_col} = {' + '.join(component_cols)}",
            "total_records": int(len(daily)),
            "valid_records": int(len(daily)),
            "consistent_records": int(consistent.sum()),
            "inconsistent_records": int((~consistent).sum()),
            "inconsistent_ratio": float((~consistent).mean()),
            "max_abs_error": float(diff.abs().max()),
            "mean_abs_error": float(diff.abs().mean()),
        })
    return pd.DataFrame(rows)


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


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """法定假期、调休、假期位置与月内工作日特征。"""
    df = df.copy()
    calendar = pd.DataFrame(
        [calendar_feature_dict(date) for date in df["date"]], index=df.index
    )
    df[calendar.columns] = calendar
    # 兼容已有模型和展示字段。
    df["days_to_next_holiday"] = df["days_to_holiday_start"]
    df["days_from_last_holiday"] = df["days_from_holiday_end"]
    df["is_day_before_holiday"] = df["is_pre_holiday_1d"]
    df["is_day_after_holiday"] = df["is_post_holiday_1d"]
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
        # 过去同星期几的多尺度统计，全部至少滞后7天。
        weekday_lags = {}
        for week in range(1, 13):
            weekday_lags[week] = df[target].shift(7 * week)
        for week in [1, 2, 3, 4]:
            df[f"{target}_same_weekday_w{week}"] = weekday_lags[week]
        for window in [4, 8, 12]:
            values = pd.concat(
                [weekday_lags[week] for week in range(1, window + 1)], axis=1
            )
            df[f"{target}_weekday_mean_{window}w"] = values.mean(axis=1)
            df[f"{target}_weekday_median_{window}w"] = values.median(axis=1)
            df[f"{target}_weekday_std_{window}w"] = values.std(axis=1)
        # 兼容原字段名。
        df[f"{target}_mean_same_weekday_4w"] = df[f"{target}_weekday_mean_4w"]
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

    quality_report = check_balance_consistency(user_balance)

    # 1. 聚合每日申购/赎回总额和用户行为结构
    user_balance["date"] = parse_date(user_balance["report_date"])
    amount_cols = [
        "total_purchase_amt", "total_redeem_amt", "consume_amt", "transfer_amt",
        "purchase_bank_amt",
    ]
    for col in amount_cols:
        if col not in user_balance.columns:
            user_balance[col] = 0.0
        user_balance[col] = pd.to_numeric(user_balance[col], errors="coerce").fillna(0.0)

    # 参考优胜方案按大/小用户分解资金流。分类仅使用截至当天的累计历史，
    # 不使用用户在未来日期的行为，保证滚动回测时不存在未来信息泄漏。
    large_user_threshold = 5_000_000
    if "user_id" in user_balance.columns:
        user_balance = user_balance.sort_values(["user_id", "date"])
        historical_max_redeem = user_balance.groupby("user_id", sort=False)["total_redeem_amt"].cummax()
    else:
        historical_max_redeem = user_balance["total_redeem_amt"]
    user_balance["is_large_user"] = (historical_max_redeem >= large_user_threshold).astype(int)

    purchase_positive = user_balance["total_purchase_amt"] > 0
    redeem_positive = user_balance["total_redeem_amt"] > 0
    user_balance["transacting_user_count"] = (purchase_positive | redeem_positive).astype(int)
    user_balance["purchase_user_count"] = purchase_positive.astype(int)
    user_balance["redeem_user_count"] = redeem_positive.astype(int)
    user_balance["both_user_count"] = (purchase_positive & redeem_positive).astype(int)
    active = purchase_positive | redeem_positive
    user_balance["large_active_user_count"] = (active & user_balance["is_large_user"].eq(1)).astype(int)
    user_balance["small_active_user_count"] = (active & user_balance["is_large_user"].eq(0)).astype(int)
    for flow in ["purchase", "redeem"]:
        source = f"total_{flow}_amt"
        user_balance[f"large_user_{flow}"] = user_balance[source].where(user_balance["is_large_user"].eq(1), 0)
        user_balance[f"small_user_{flow}"] = user_balance[source].where(user_balance["is_large_user"].eq(0), 0)

    # 交易级阈值特征；原始金额单位与赛题提交口径一致，不做擅自换算。
    user_balance["large_purchase_user_count"] = (user_balance["total_purchase_amt"] >= 1_000_000).astype(int)
    user_balance["large_redeem_user_count"] = (user_balance["total_redeem_amt"] >= 1_000_000).astype(int)
    user_balance["large_purchase_amt"] = user_balance["total_purchase_amt"].where(
        user_balance["total_purchase_amt"] >= 1_000_000, 0
    )
    user_balance["large_redeem_amt"] = user_balance["total_redeem_amt"].where(
        user_balance["total_redeem_amt"] >= 1_000_000, 0
    )
    user_balance["very_large_purchase_amt"] = user_balance["total_purchase_amt"].where(
        user_balance["total_purchase_amt"] >= 5_000_000, 0
    )
    user_balance["very_large_redeem_amt"] = user_balance["total_redeem_amt"].where(
        user_balance["total_redeem_amt"] >= 5_000_000, 0
    )
    daily = (
        user_balance.groupby("date", as_index=False)
        .agg(
            purchase=("total_purchase_amt", "sum"),
            redeem=("total_redeem_amt", "sum"),
            consume_amt=("consume_amt", "sum"),
            transfer_amt=("transfer_amt", "sum"),
            large_user_purchase=("large_user_purchase", "sum"),
            small_user_purchase=("small_user_purchase", "sum"),
            large_user_redeem=("large_user_redeem", "sum"),
            small_user_redeem=("small_user_redeem", "sum"),
            user_count=("user_id", "nunique") if "user_id" in user_balance.columns else ("report_date", "count"),
            record_count=("report_date", "count"),
            transacting_user_count=("transacting_user_count", "sum"),
            purchase_user_count=("purchase_user_count", "sum"),
            redeem_user_count=("redeem_user_count", "sum"),
            both_user_count=("both_user_count", "sum"),
            large_active_user_count=("large_active_user_count", "sum"),
            small_active_user_count=("small_active_user_count", "sum"),
            large_purchase_user_count=("large_purchase_user_count", "sum"),
            large_redeem_user_count=("large_redeem_user_count", "sum"),
            large_purchase_amt=("large_purchase_amt", "sum"),
            large_redeem_amt=("large_redeem_amt", "sum"),
            very_large_purchase_amt=("very_large_purchase_amt", "sum"),
            very_large_redeem_amt=("very_large_redeem_amt", "sum"),
            purchase_bank_amt=("purchase_bank_amt", "sum"),
        )
        .sort_values("date")
    )
    active_denominator = daily["transacting_user_count"].clip(lower=1)
    daily["purchase_per_active_user"] = daily["purchase"] / active_denominator
    daily["redeem_per_active_user"] = daily["redeem"] / active_denominator
    daily["consume_per_active_user"] = daily["consume_amt"] / active_denominator
    daily["transfer_per_active_user"] = daily["transfer_amt"] / active_denominator
    daily["large_purchase_per_active_user"] = daily["large_user_purchase"] / daily["large_active_user_count"].clip(lower=1)
    daily["small_purchase_per_active_user"] = daily["small_user_purchase"] / daily["small_active_user_count"].clip(lower=1)
    daily["large_redeem_per_active_user"] = daily["large_user_redeem"] / daily["large_active_user_count"].clip(lower=1)
    daily["small_redeem_per_active_user"] = daily["small_user_redeem"] / daily["small_active_user_count"].clip(lower=1)
    daily["purchase_bank_share"] = daily["purchase_bank_amt"] / (daily["purchase"] + 1)
    daily["transfer_redeem_share"] = daily["transfer_amt"] / (daily["redeem"] + 1)

    quality_report = pd.concat(
        [quality_report, check_component_consistency(daily)], ignore_index=True, sort=False
    )
    DATA_QUALITY_REPORT_PATH.parent.mkdir(exist_ok=True)
    quality_report.to_csv(DATA_QUALITY_REPORT_PATH, index=False, encoding="utf-8-sig")
    print(f"数据质量报告已生成：{DATA_QUALITY_REPORT_PATH}")

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
    daily = add_behavior_history_features(daily)

    # 5. 简单缺失处理
    daily = daily.sort_values("date")
    numeric_cols = daily.select_dtypes(include=[np.number]).columns
    # 时间序列只允许向前填充；bfill 会把未来观测传播到过去。
    daily[numeric_cols] = daily[numeric_cols].ffill()
    # 剩余 NaN 用 0 填充（如 roll_std 在窗口不足时）
    daily[numeric_cols] = daily[numeric_cols].fillna(0)

    DAILY_FEATURES_PATH.parent.mkdir(exist_ok=True)
    daily.to_csv(DAILY_FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(f"已生成特征文件：{DAILY_FEATURES_PATH}")
    print(f"样本数量：{len(daily)}，字段数量：{len(daily.columns)}")


if __name__ == "__main__":
    main()
