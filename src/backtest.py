"""Expanding-window monthly backtest with recursive multi-step forecasts."""

import argparse

import numpy as np
import pandas as pd

from .config import (
    BACKTEST_METRICS_PATH,
    BACKTEST_PRED_PATH,
    DAILY_FEATURES_PATH,
    PURCHASE_MODEL_WEIGHT,
    REDEEM_MODEL_WEIGHT,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PRESET,
)
from .evaluate import evaluate_prediction
from .predict import build_future_row, weekday_anchor
from .train import build_lightgbm, fit_single_model, get_feature_columns
from .calendar_features import CALENDAR_FEATURE_COLUMNS


LAG_ROLL_PREFIXES = (
    "purchase_lag_", "redeem_lag_", "purchase_roll_", "redeem_roll_",
    "purchase_same_", "redeem_same_", "purchase_diff_", "redeem_diff_",
    "purchase_weekday_", "redeem_weekday_",
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
} | CALENDAR_FEATURE_COLUMNS

STAGE_MONTHS = {
    "development": ["2014-04", "2014-05", "2014-06", "2014-07"],
    "confirm": ["2014-08"],
    "all": ["2014-04", "2014-05", "2014-06", "2014-07", "2014-08"],
}


class MeanModel:
    """Prediction-only ensemble used by the backtest."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([model.predict(X) for model in self.models], axis=0)


def build_xgboost(seed=42, **overrides):
    from xgboost import XGBRegressor

    params = {
        "n_estimators": 3000,
        "learning_rate": 0.01,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": 100,
    }
    params.update(overrides)
    return XGBRegressor(**params)


def build_ridge(alpha=10.0):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


MODEL_PRESETS = {
    "default": {},
    "conservative": {
        "lightgbm": {
            "n_estimators": 4000, "learning_rate": 0.005,
            "num_leaves": 15, "max_depth": 5, "min_child_samples": 10,
            "reg_alpha": 0.1, "reg_lambda": 0.2,
        },
        "xgboost": {
            "n_estimators": 2500, "learning_rate": 0.01,
            "max_depth": 3, "min_child_weight": 10,
            "reg_alpha": 0.1, "reg_lambda": 2.0,
        },
    },
    "shallow": {
        "lightgbm": {
            "n_estimators": 3000, "learning_rate": 0.005,
            "num_leaves": 7, "max_depth": 3, "min_child_samples": 12,
            "reg_alpha": 0.1, "reg_lambda": 0.5,
        },
        "xgboost": {
            "n_estimators": 2000, "learning_rate": 0.02,
            "max_depth": 2, "min_child_weight": 8,
            "reg_alpha": 0.1, "reg_lambda": 2.0,
        },
    },
}


def model_overrides(model_name, preset):
    return MODEL_PRESETS[preset].get(model_name, {})


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


def make_training_target(frame, target, target_mode):
    if target_mode == "direct":
        return np.log1p(frame[target])
    if target_mode == "residual":
        baseline = frame[f"{target}_weekday_mean_12w"].clip(lower=1)
        return np.log((frame[target] + 1) / (baseline + 1))
    raise ValueError(f"不支持的目标模式：{target_mode}")


def train_refitted_ensemble(
    history, feature_cols, target, n_seeds, target_mode="direct",
    model_name="lightgbm", preset="default", model_params=None,
    start_seed=42,
):
    """Tune on the last historical month, then refit on all fold history."""
    last_period = history["date"].dt.to_period("M").max()
    tune = history[history["date"].dt.to_period("M") == last_period]
    fit = history[history["date"].dt.to_period("M") < last_period]
    if fit.empty or tune.empty:
        raise ValueError("回测历史不足以划分训练集和早停集")

    X_fit = fit[feature_cols].fillna(0)
    X_tune = tune[feature_cols].fillna(0)
    y_fit = make_training_target(fit, target, target_mode)
    y_tune = make_training_target(tune, target, target_mode)
    X_full = history[feature_cols].fillna(0)
    y_full = make_training_target(history, target, target_mode)

    if model_name == "ridge":
        model = build_ridge(**(model_params or {}))
        model.fit(X_full, y_full)
        return MeanModel([model]), [0]

    models = []
    overrides = {**model_overrides(model_name, preset), **(model_params or {})}
    best_iterations = []
    for seed in range(start_seed, start_seed + n_seeds):
        if model_name == "xgboost":
            tuned = build_xgboost(seed=seed, **overrides)
            tuned.fit(X_fit, y_fit, eval_set=[(X_tune, y_tune)], verbose=False)
            best_iteration = int(getattr(tuned, "best_iteration", 2999)) + 1
            final_overrides = {
                **overrides,
                "n_estimators": best_iteration,
                "early_stopping_rounds": None,
            }
            final_model = build_xgboost(seed=seed, **final_overrides)
        else:
            tuned = fit_single_model(
                build_lightgbm(seed=seed, **overrides), X_fit, y_fit, X_tune, y_tune,
                use_early_stop=True,
            )
            best_iteration = int(getattr(tuned, "best_iteration_", 0) or 10000)
            final_model = build_lightgbm(seed=seed, **overrides)
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
    purchase_model_weight=1.0,
    redeem_model_weight=1.0,
    smooth_weight=0.99,
    target_mode="direct",
):
    feature_history = history.copy()
    history = history[["date", "purchase", "redeem"]].copy()
    anchor_history = history.copy()
    p_lo, p_hi = np.percentile(history["purchase"], [1, 99])
    r_lo, r_hi = np.percentile(history["redeem"], [1, 99])
    rows = []

    for date in forecast_dates:
        p_anchor = weekday_anchor(date, anchor_history, "purchase")
        r_anchor = weekday_anchor(date, anchor_history, "redeem")
        X = build_future_row(date, history, feature_cols, exog_values, feature_history)
        p_raw = float(model_purchase.predict(X)[0])
        r_raw = float(model_redeem.predict(X)[0])
        if target_mode == "residual":
            p_pred = float((p_anchor + 1) * np.exp(p_raw) - 1)
            r_pred = float((r_anchor + 1) * np.exp(r_raw) - 1)
        else:
            p_pred = float(np.expm1(p_raw))
            r_pred = float(np.expm1(r_raw))
        p_pred = smooth_weight * p_pred + (1 - smooth_weight) * float(history["purchase"].tail(7).mean())
        r_pred = smooth_weight * r_pred + (1 - smooth_weight) * float(history["redeem"].tail(7).mean())

        if apply_month_end_boost and date.day >= date.days_in_month - 2:
            p_pred *= 1.2
            r_pred *= 1.3

        p_recursive = float(np.clip(p_pred, p_lo, p_hi))
        r_recursive = float(np.clip(r_pred, r_lo, r_hi))

        # 仅使用预测月开始前的真实历史构造稳定星期周期锚点，避免递归误差污染。
        p_pred = purchase_model_weight * p_recursive + (1 - purchase_model_weight) * p_anchor
        r_pred = redeem_model_weight * r_recursive + (1 - redeem_model_weight) * r_anchor

        p_pred = float(np.clip(p_pred, p_lo, p_hi))
        r_pred = float(np.clip(r_pred, r_lo, r_hi))
        rows.append({"date": date, "purchase_pred": p_pred, "redeem_pred": r_pred})
        history = pd.concat([
            history,
            pd.DataFrame({
                "date": [date], "purchase": [p_recursive], "redeem": [r_recursive]
            }),
        ], ignore_index=True)
    return pd.DataFrame(rows)


def evaluate_fold(
    df, feature_cols, period, n_seeds, target_mode="direct",
    model_name="lightgbm", preset="default"
):
    fold_start = period.start_time
    fold_end = period.end_time.normalize()
    history = df[df["date"] < fold_start].copy()
    actual = df[(df["date"] >= fold_start) & (df["date"] <= fold_end)].copy()
    if actual.empty:
        raise ValueError(f"回测月份 {period} 没有真实数据")

    print(f"[{period}] 训练历史截至 {history['date'].max().date()}，共 {len(history)} 天")
    model_purchase, p_iters = train_refitted_ensemble(
        history, feature_cols, "purchase", n_seeds, target_mode, model_name, preset
    )
    model_redeem, r_iters = train_refitted_ensemble(
        history, feature_cols, "redeem", n_seeds, target_mode, model_name, preset
    )
    exog_values = build_exogenous_values(df, feature_cols, period)

    prediction_frames = []
    metric_rows = []
    if target_mode == "residual":
        strategies = [
            ("residual_lgbm", False, 1.0, 1.0),
            ("residual_blend", False, PURCHASE_MODEL_WEIGHT, REDEEM_MODEL_WEIGHT),
        ]
    else:
        strategies = [
            ("base", False, 1.0, 1.0),
            ("seasonal_blend", False, PURCHASE_MODEL_WEIGHT, REDEEM_MODEL_WEIGHT),
            ("month_end_boost", True, 1.0, 1.0),
        ]
    for strategy, use_boost, purchase_weight, redeem_weight in strategies:
        pred = recursive_forecast(
            model_purchase, model_redeem, history,
            pd.date_range(fold_start, fold_end, freq="D"),
            feature_cols, exog_values, apply_month_end_boost=use_boost,
            purchase_model_weight=purchase_weight,
            redeem_model_weight=redeem_weight,
            target_mode=target_mode,
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
        "--stage", choices=list(STAGE_MONTHS), default="all",
        help="development用于参数筛选；confirm仅评估8月；all用于冻结方案后的完整报告",
    )
    parser.add_argument(
        "--months", nargs="+",
        help="自定义回测月份；提供后覆盖 --stage 的默认月份",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=1,
        help="每个目标的随机种子数；快速回测默认1，完整集成使用3",
    )
    parser.add_argument(
        "--target-mode", choices=["direct", "residual"], default="direct",
        help="direct直接预测金额；residual预测相对12周星期基线的对数残差",
    )
    parser.add_argument(
        "--model", choices=["lightgbm", "xgboost", "ridge"], default=DEFAULT_MODEL_NAME,
        help="滚动回测使用的回归模型",
    )
    parser.add_argument(
        "--preset", choices=list(MODEL_PRESETS), default=DEFAULT_MODEL_PRESET,
        help="模型参数预设，用于小样本稳健性比较",
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
    month_values = args.months or STAGE_MONTHS[args.stage]
    evaluation_stage = "custom" if args.months else args.stage
    periods = [pd.Period(value, freq="M") for value in month_values]

    prediction_frames = []
    metric_rows = []
    for period in periods:
        fold_predictions, fold_metrics = evaluate_fold(
            df, feature_cols, period, args.n_seeds, args.target_mode,
            args.model, args.preset
        )
        prediction_frames.extend(fold_predictions)
        for row in fold_metrics:
            row["evaluation_stage"] = evaluation_stage
        metric_rows.extend(fold_metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    for strategy, group in predictions.groupby("strategy"):
        overall = evaluate_prediction(group)
        strategy_folds = metrics[metrics["strategy"] == strategy]
        worst_index = strategy_folds["risk_adjusted_loss"].idxmax()
        worst_fold = strategy_folds.loc[worst_index]
        metrics = pd.concat([
            metrics,
            pd.DataFrame([{
                "fold": "overall", "strategy": strategy,
                "evaluation_stage": evaluation_stage, **overall,
                "worst_fold": str(worst_fold["fold"]),
                "worst_fold_risk_adjusted_loss": float(worst_fold["risk_adjusted_loss"]),
                "fold_risk_adjusted_loss_std": float(strategy_folds["risk_adjusted_loss"].std(ddof=0)),
                "worst_fold_flow_mape": float(strategy_folds["flow_mape"].max()),
                "fold_flow_mape_std": float(strategy_folds["flow_mape"].std(ddof=0)),
                "purchase_zero_score_days": int((group["purchase_error"] > 0.3).sum()),
                "redeem_zero_score_days": int((group["redeem_error"] > 0.3).sum()),
                "purchase_best_iterations": "", "redeem_best_iterations": "",
            }]),
        ], ignore_index=True)

    BACKTEST_PRED_PATH.parent.mkdir(exist_ok=True)
    predictions.to_csv(BACKTEST_PRED_PATH, index=False, encoding="utf-8-sig")
    metrics.to_csv(BACKTEST_METRICS_PATH, index=False, encoding="utf-8-sig")
    print("\n回测汇总：")
    print(metrics[[
        "fold", "strategy", "flow_mape", "tail_p90",
        "severe_30_ratio", "risk_adjusted_loss", "total_score",
    ]].to_string(index=False))
    print(f"\n逐日结果：{BACKTEST_PRED_PATH}")
    print(f"指标汇总：{BACKTEST_METRICS_PATH}")


if __name__ == "__main__":
    main()
