"""Leakage-safe historical features derived from daily user behavior aggregates."""

import numpy as np
import pandas as pd


BEHAVIOR_BASE_COLUMNS = (
    "user_count",
    "transacting_user_count",
    "purchase_user_count",
    "redeem_user_count",
    "both_user_count",
    "large_purchase_user_count",
    "large_redeem_user_count",
    "large_purchase_amt",
    "large_redeem_amt",
    "very_large_purchase_amt",
    "very_large_redeem_amt",
    "purchase_bank_share",
    "transfer_redeem_share",
)


def add_behavior_history_features(df):
    """Create features that only use observations strictly before each row."""
    df = df.sort_values("date").copy()
    for column in BEHAVIOR_BASE_COLUMNS:
        if column not in df.columns:
            continue
        shifted = df[column].shift(1)
        for window in (7, 30):
            df[f"behavior_{column}_roll_mean_{window}"] = shifted.rolling(window).mean()
        weekday_lags = [df[column].shift(7 * week) for week in range(1, 13)]
        df[f"behavior_{column}_weekday_mean_4w"] = pd.concat(
            weekday_lags[:4], axis=1
        ).mean(axis=1)
        df[f"behavior_{column}_weekday_mean_12w"] = pd.concat(
            weekday_lags, axis=1
        ).mean(axis=1)
    return df


def behavior_feature_dict(date, history):
    """Build future behavior features from pre-forecast aggregate history."""
    date = pd.Timestamp(date)
    features = {}
    for column in BEHAVIOR_BASE_COLUMNS:
        if column not in history.columns:
            continue
        values = history[column].dropna()
        if values.empty:
            continue
        for window in (7, 30):
            features[f"behavior_{column}_roll_mean_{window}"] = float(
                values.tail(window).mean()
            )
        same_weekday = history.loc[
            history["date"].dt.dayofweek == date.dayofweek, column
        ].dropna()
        fallback = float(values.tail(30).mean())
        features[f"behavior_{column}_weekday_mean_4w"] = float(
            same_weekday.tail(4).mean()
        ) if not same_weekday.empty else fallback
        features[f"behavior_{column}_weekday_mean_12w"] = float(
            same_weekday.tail(12).mean()
        ) if not same_weekday.empty else fallback
    return features
