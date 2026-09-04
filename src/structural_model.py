"""可解释的非递归结构化时间序列模型。

在 log1p 空间把金额拆为长期水平、星期周期、标准化月内位置和节假日效应。
这与乘法分解等价，但无需递归使用未来预测值，因此不会累积 30 天误差。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar_features import calendar_feature_dict


def month_position(date) -> int:
    """把不同长度月份映射到 0..29，替代删除二月或手工把31日并入30日。"""
    date = pd.Timestamp(date)
    if date.days_in_month <= 1:
        return 0
    return int(round((date.day - 1) * 29 / (date.days_in_month - 1)))


def holiday_type(date) -> str:
    feature = calendar_feature_dict(date)
    if feature["is_holiday"]:
        return "holiday"
    if feature["is_special_workday"]:
        return "special_workday"
    if feature["is_pre_holiday_1d"]:
        return "pre_holiday"
    if feature["is_post_holiday_1d"]:
        return "post_holiday"
    return "normal"


@dataclass
class StructuralParams:
    trend_window: int = 120
    level_window: int = 28
    seasonal_window: int = 365
    trend_strength: float = 0.5
    max_daily_log_slope: float = 0.006
    weekday_shrinkage: float = 8.0
    month_shrinkage: float = 15.0
    holiday_shrinkage: float = 8.0
    clip_lower: float = 0.01
    clip_upper: float = 0.99


class StructuralSeriesForecaster:
    def __init__(self, params=None):
        self.params = params or StructuralParams()

    @staticmethod
    def _shrunk_effect(residual, groups, shrinkage):
        frame = pd.DataFrame({"residual": residual, "group": groups}).dropna()
        stats = frame.groupby("group")["residual"].agg(["mean", "count"])
        effect = stats["mean"] * stats["count"] / (stats["count"] + shrinkage)
        if not effect.empty:
            weights = stats["count"] / stats["count"].sum()
            effect = effect - float((effect * weights).sum())
        return effect.to_dict()

    def fit(self, dates, values):
        frame = pd.DataFrame({"date": pd.to_datetime(dates), "value": values}).dropna()
        frame = frame.sort_values("date")
        if frame.empty:
            raise ValueError("结构模型没有可用训练样本")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce").fillna(0).clip(lower=0)
        frame["log_value"] = np.log1p(frame["value"])
        frame["t"] = (frame["date"] - frame["date"].min()).dt.days.astype(float)

        recent = frame.tail(max(14, self.params.trend_window)).copy()
        smooth = recent["log_value"].rolling(7, center=True, min_periods=3).median()
        valid = smooth.notna()
        if valid.sum() >= 2:
            weights = np.geomspace(0.35, 1.0, valid.sum())
            slope = np.polyfit(recent.loc[valid, "t"], smooth[valid], 1, w=weights)[0]
        else:
            slope = 0.0
        slope = float(np.clip(
            slope * self.params.trend_strength,
            -self.params.max_daily_log_slope,
            self.params.max_daily_log_slope,
        ))
        last_t = float(frame["t"].iloc[-1])
        level_source = frame.tail(max(7, self.params.level_window))
        last_level = float(np.median(level_source["log_value"] - slope * (level_source["t"] - last_t)))
        trend = last_level + slope * (frame["t"] - last_t)

        seasonal = frame.tail(max(60, self.params.seasonal_window)).copy()
        residual = seasonal["log_value"] - trend.loc[seasonal.index]
        weekday = seasonal["date"].dt.dayofweek
        self.weekday_effect = self._shrunk_effect(residual, weekday, self.params.weekday_shrinkage)
        residual = residual - weekday.map(self.weekday_effect).fillna(0)
        positions = seasonal["date"].map(month_position)
        self.month_effect = self._shrunk_effect(residual, positions, self.params.month_shrinkage)
        residual = residual - positions.map(self.month_effect).fillna(0)
        holidays = seasonal["date"].map(holiday_type)
        self.holiday_effect = self._shrunk_effect(residual, holidays, self.params.holiday_shrinkage)

        clip_history = frame.tail(max(60, self.params.seasonal_window))["value"]
        self.lower = float(clip_history.quantile(self.params.clip_lower))
        self.upper = float(clip_history.quantile(self.params.clip_upper))
        self.origin = frame["date"].min()
        self.last_t = last_t
        self.last_level = last_level
        self.slope = slope
        return self

    def predict(self, dates):
        dates = pd.Series(pd.to_datetime(dates))
        t = (dates - self.origin).dt.days.astype(float)
        log_prediction = self.last_level + self.slope * (t - self.last_t)
        log_prediction += dates.dt.dayofweek.map(self.weekday_effect).fillna(0)
        log_prediction += dates.map(month_position).map(self.month_effect).fillna(0)
        log_prediction += dates.map(holiday_type).map(self.holiday_effect).fillna(0)
        prediction = np.expm1(log_prediction).clip(lower=0)
        return np.clip(prediction.to_numpy(), self.lower, self.upper)


class StructuralFlowForecaster:
    """按不同业务分解口径预测，再恢复申购和赎回总额。"""

    MODES = ("total", "user_segment", "flow_component", "per_active_user")

    def __init__(self, mode="total", params=None):
        if mode not in self.MODES:
            raise ValueError(f"未知结构模式：{mode}")
        self.mode = mode
        self.params = params or StructuralParams()
        self.models = {}

    def _targets(self):
        if self.mode == "total":
            return ["purchase", "redeem"]
        if self.mode == "user_segment":
            return ["large_user_purchase", "small_user_purchase", "large_user_redeem", "small_user_redeem"]
        if self.mode == "flow_component":
            return ["purchase", "consume_amt", "transfer_amt"]
        return ["transacting_user_count", "purchase_per_active_user", "consume_per_active_user", "transfer_per_active_user"]

    def fit(self, history):
        missing = [column for column in self._targets() if column not in history.columns]
        if missing:
            raise ValueError(f"特征文件缺少结构模型字段：{', '.join(missing)}；请重新运行预处理")
        for target in self._targets():
            self.models[target] = StructuralSeriesForecaster(self.params).fit(history["date"], history[target])
        return self

    def predict(self, dates):
        dates = pd.DatetimeIndex(dates)
        components = {name: model.predict(dates) for name, model in self.models.items()}
        if self.mode == "total":
            purchase, redeem = components["purchase"], components["redeem"]
        elif self.mode == "user_segment":
            purchase = components["large_user_purchase"] + components["small_user_purchase"]
            redeem = components["large_user_redeem"] + components["small_user_redeem"]
        elif self.mode == "flow_component":
            purchase = components["purchase"]
            redeem = components["consume_amt"] + components["transfer_amt"]
        else:
            users = components["transacting_user_count"]
            purchase = users * components["purchase_per_active_user"]
            redeem = users * (components["consume_per_active_user"] + components["transfer_per_active_user"])
        result = pd.DataFrame({"date": dates, "purchase_pred": purchase, "redeem_pred": redeem})
        for name, values in components.items():
            result[f"component_{name}"] = values
        return result
