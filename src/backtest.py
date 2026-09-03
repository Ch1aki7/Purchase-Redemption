"""Expanding-window monthly backtest with recursive multi-step forecasts."""

import argparse

import numpy as np
import pandas as pd

from .config import BACKTEST_METRICS_PATH, BACKTEST_PRED_PATH, DAILY_FEATURES_PATH
from .evaluate import evaluate_prediction
from .predict import build_future_row
from .train import build_lightgbm, fit_single_model, get_feature_columns


LAG_ROLL_PREFIXES = (
    "purchase_lag_", "redeem_lag_", "purchase_roll_", "redeem_roll_",
    "purchase_same_", "redeem_same_", "purchase_diff_", "redeem_diff_",
    "net_inflow", "purchase_redeem_ratio",
)

TIME_COLUMNS = {
    "dayofweek", "dayofmonth", "month", "is_weekend", "is_month_start",
    "is_month_end", "dayofweek_sin", "dayofweek_cos", "dayofmonth_sin",
    "dayofmonth_cos", "month_sin", "month_cos", "days_to_month_start",
    "days_to_month_end", "is_first_3_days", "is_last_3_days",
    "is_first_7_days", "is_last_7_days", "month_start_weight",
    "month_end_weight", "is_holiday", "days_to_next_holiday",
    "days_from_last_holiday", "is_day_before_holiday", "is_day_after_holiday",
}


class MeanModel:
    """Prediction-only ensemble used by the backtest."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([model.predict(X) for model in self.models], axis=0)


def select_exogenous_columns(feature_cols):
    return [
        c for c in feature_cols
        if c not in TIME_COLUMNS and not c.startswith(LAG_ROLL_PREFIXES)
    ]


def build_exogenous_values(df, feature_cols, forecast_period):
    source_period = forecast_period - 1
    source = df[df["date"].dt.to_period("M") == source_period]
    if source.empty:
        raise ValueError(f"无法为 {forecast_period} 构造外部变量：缺少 {source_period} 数据")
    return {
        c: float(source[c].mean())
        for c in select_exogenous_columns(feature_cols)
        if c in source.columns
    }


def train_refitted_ensemble(history, feature_cols, target, n_seeds):
    """Tune on the last historical month, then refit on all fold history."""
    last_period = history["date"].dt.to_period("M").max()
    tune = history[history["date"].dt.to_period("M") == last_period]
    fit = history[history["date"].dt.to_period("M") < last_period]
    if fit.empty or tune.empty:
        raise ValueError("回测历史不足以划分训练集和早停集")

    X_fit = fit[feature_cols].fillna(0)
    X_tune = tune[feature_cols].fillna(0)
    y_fit = np.log1p(fit[target])
    y_tune = np.log1p(tune[target])
    X_full = history[feature_cols].fillna(0)
    y_full = np.log1p(history[target])

    models = []
    best_iterations = []
    for seed in range(42, 42 + n_seeds):
        tuned = fit_single_model(
            build_lightgbm(seed=seed), X_fit, y_fit, X_tune, y_tune,
            use_early_stop=True,
        )
        best_iteration = int(getattr(tuned, "best_iteration_", 0) or 10000)
        final_model = build_lightgbm(seed=seed)
        final_model.set_params(n_estimators=best_iteration)
        final_model.fit(X_full, y_full)
        models.append(final_model)
        best_iterations.append(best_iteration)
    return MeanModel(models), best_iterations


def recursive_forecast(
    model_purchase,
    model_redeem,
    history,
    forecast_dates,
    feature_cols,
    exog_values,
    apply_month_end_boost,
    smooth_weight=0.99,
):
    history = history[["date", "purchase", "redeem"]].copy()
    p_lo, p_hi = np.percentile(history["purchase"], [1, 99])
    r_lo, r_hi = np.percentile(history["redeem"], [1, 99])
    rows = []

    for date in forecast_dates:
        X = build_future_row(date, history, feature_cols, exog_values, history)
        p_pred = float(np.expm1(model_purchase.predict(X)[0]))
        r_pred = float(np.expm1(model_redeem.predict(X)[0]))
        p_pred = smooth_weight * p_pred + (1 - smooth_weight) * float(history["purchase"].tail(7).mean())
        r_pred = smooth_weight * r_pred + (1 - smooth_weight) * float(history["redeem"].tail(7).mean())

        if apply_month_end_boost and date.day >= date.days_in_month - 2:
            p_pred *= 1.2
            r_pred *= 1.3

        p_pred = float(np.clip(p_pred, p_lo, p_hi))
        r_pred = float(np.clip(r_pred, r_lo, r_hi))
        rows.append({"date": date, "purchase_pred": p_pred, "redeem_pred": r_pred})
        history = pd.concat([
            history,
            pd.DataFrame({"date": [date], "purchase": [p_pred], "redeem": [r_pred]}),
        ], ignore_index=True)
    return pd.DataFrame(rows)


def evaluate_fold(df, feature_cols, period, n_seeds):
    fold_start = period.start_time
    fold_end = period.end_time.normalize()
    history = df[df["date"] < fold_start].copy()
    actual = df[(df["date"] >= fold_start) & (df["date"] <= fold_end)].copy()
    if actual.empty:
        raise ValueError(f"回测月份 {period} 没有真实数据")

    print(f"[{period}] 训练历史截至 {history['date'].max().date()}，共 {len(history)} 天")
    model_purchase, p_iters = train_refitted_ensemble(history, feature_cols, "purchase", n_seeds)
    model_redeem, r_iters = train_refitted_ensemble(history, feature_cols, "redeem", n_seeds)
    exog_values = build_exogenous_values(df, feature_cols, period)

    prediction_frames = []
    metric_rows = []
    for strategy, use_boost in [("base", False), ("month_end_boost", True)]:
        pred = recursive_forecast(
            model_purchase, model_redeem, history,
            pd.date_range(fold_start, fold_end, freq="D"),
            feature_cols, exog_values, apply_month_end_boost=use_boost,
        )
        result = pred.merge(
            actual[["date", "purchase", "redeem"]].rename(
                columns={"purchase": "purchase_true", "redeem": "redeem_true"}
            ),
            on="date",
            how="inner",
        )
        result["fold"] = str(period)
        result["strategy"] = strategy
        result["purchase_error"] = np.abs(result["purchase_pred"] - result["purchase_true"]) / result["purchase_true"].replace(0, 1)
        result["redeem_error"] = np.abs(result["redeem_pred"] - result["redeem_true"]) / result["redeem_true"].replace(0, 1)
        prediction_frames.append(result)

        metrics = evaluate_prediction(result)
        metric_rows.append({
            "fold": str(period), "strategy": strategy, **metrics,
            "purchase_zero_score_days": int((result["purchase_error"] > 0.3).sum()),
            "redeem_zero_score_days": int((result["redeem_error"] > 0.3).sum()),
            "purchase_best_iterations": "/".join(map(str, p_iters)),
            "redeem_best_iterations": "/".join(map(str, r_iters)),
        })
        print(f"[{period}] {strategy}: score={metrics['total_score']:.4f}")
    return prediction_frames, metric_rows


def parse_args():
    parser = argparse.ArgumentParser(description="按月执行扩展窗口递归滚动回测")
    parser.add_argument(
        "--months", nargs="+",
        default=["2014-04", "2014-05", "2014-06", "2014-07", "2014-08"],
        help="回测月份，例如 --months 2014-06 2014-07 2014-08",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=1,
        help="每个目标的随机种子数；快速回测默认1，完整集成使用3",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_seeds < 1:
        raise ValueError("--n-seeds 必须大于等于1")
    if not DAILY_FEATURES_PATH.exists():
        raise FileNotFoundError("缺少 daily_features.csv，请先运行 python -m src.preprocess")

    df = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] >= pd.Timestamp("2013-08-01")].copy()
    feature_cols = get_feature_columns(df)
    periods = [pd.Period(value, freq="M") for value in args.months]

    prediction_frames = []
    metric_rows = []
    for period in periods:
        fold_predictions, fold_metrics = evaluate_fold(df, feature_cols, period, args.n_seeds)
        prediction_frames.extend(fold_predictions)
        metric_rows.extend(fold_metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    for strategy, group in predictions.groupby("strategy"):
        overall = evaluate_prediction(group)
        metrics = pd.concat([
            metrics,
            pd.DataFrame([{
                "fold": "overall", "strategy": strategy, **overall,
                "purchase_zero_score_days": int((group["purchase_error"] > 0.3).sum()),
                "redeem_zero_score_days": int((group["redeem_error"] > 0.3).sum()),
                "purchase_best_iterations": "", "redeem_best_iterations": "",
            }]),
        ], ignore_index=True)

    BACKTEST_PRED_PATH.parent.mkdir(exist_ok=True)
    predictions.to_csv(BACKTEST_PRED_PATH, index=False, encoding="utf-8-sig")
    metrics.to_csv(BACKTEST_METRICS_PATH, index=False, encoding="utf-8-sig")
    print("\n回测汇总：")
    print(metrics[["fold", "strategy", "purchase_mape", "redeem_mape", "total_score"]].to_string(index=False))
    print(f"\n逐日结果：{BACKTEST_PRED_PATH}")
    print(f"指标汇总：{BACKTEST_METRICS_PATH}")


if __name__ == "__main__":
    main()
