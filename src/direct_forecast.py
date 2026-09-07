"""Non-recursive multi-horizon forecasting experiments.

Each training row represents an origin date and a forecast horizon.  Features use
only information known at the origin, so a single model can predict horizons 1-30
without feeding predictions back into later horizons.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import (
    BACKTEST_PRED_PATH,
    DAILY_FEATURES_PATH,
    DIRECT_BACKTEST_METRICS_PATH,
    DIRECT_BACKTEST_PRED_PATH,
    DIRECT_CANDIDATE_PATH,
    RESIDUAL_CANDIDATE_PATH,
    SHAPE_BACKTEST_METRICS_PATH,
    SHAPE_CANDIDATE_PATH,
    SUBMISSION_PATH,
)
from .evaluate import evaluate_prediction
from .predict import date_features, holiday_features, weekday_seasonal_forecast


MAX_HORIZON = 30
MIN_HISTORY = 84
RESIDUAL_BLEND_PARAMS = {
    "purchase": {"direct_weight": 0.10, "scale": 1.0125},
    "redeem": {"direct_weight": 0.20, "scale": 1.0125},
}
SHAPE_ALPHA_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
SHAPE_MAX_DAILY_CHANGE = 0.03


def _history_features(df: pd.DataFrame, origin: int) -> dict[str, float]:
    """Build features available at the close of ``origin``."""
    row: dict[str, float] = {}
    for target in ("purchase", "redeem"):
        values = df[target].iloc[: origin + 1].to_numpy(dtype=float)
        row[f"{target}_last"] = values[-1]
        for lag in (7, 14, 28, 56):
            row[f"{target}_lag_{lag}"] = values[-lag] if len(values) >= lag else values[0]
        for window in (7, 14, 28, 56):
            recent = values[-window:]
            row[f"{target}_mean_{window}"] = float(np.mean(recent))
            row[f"{target}_median_{window}"] = float(np.median(recent))
            row[f"{target}_std_{window}"] = float(np.std(recent, ddof=1))

        recent28 = float(np.mean(values[-28:]))
        prior28 = float(np.mean(values[-56:-28]))
        row[f"{target}_level_ratio"] = recent28 / prior28 if prior28 > 0 else 1.0
        x = np.arange(min(28, len(values)), dtype=float)
        y = values[-len(x):]
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0
        row[f"{target}_trend_28"] = slope / max(recent28, 1.0)

        dates = df["date"].iloc[: origin + 1]
        for weeks in (4, 8, 12):
            for weekday in range(7):
                same_weekday = values[dates.dt.dayofweek.to_numpy() == weekday]
                row[f"{target}_weekday_{weekday}_mean_{weeks}"] = float(
                    np.mean(same_weekday[-weeks:])
                )
    row["net_inflow_last"] = row["purchase_last"] - row["redeem_last"]
    row["purchase_redeem_ratio"] = row["purchase_last"] / max(row["redeem_last"], 1.0)
    return row


def _forecast_date_features(date: pd.Timestamp, horizon: int) -> dict[str, float]:
    row = {f"forecast_{key}": value for key, value in date_features(date).items()}
    row.update({f"forecast_{key}": value for key, value in holiday_features(date).items()})
    row["horizon"] = horizon
    row["horizon_sqrt"] = float(np.sqrt(horizon))
    row["horizon_week"] = (horizon - 1) // 7
    return row


def build_training_matrix(df: pd.DataFrame, max_horizon: int = MAX_HORIZON):
    records = []
    purchase_targets = []
    redeem_targets = []
    target_dates = []
    for origin in range(MIN_HISTORY - 1, len(df) - 1):
        shared = _history_features(df, origin)
        available = min(max_horizon, len(df) - origin - 1)
        for horizon in range(1, available + 1):
            target_index = origin + horizon
            target_date = pd.Timestamp(df["date"].iloc[target_index])
            records.append({**shared, **_forecast_date_features(target_date, horizon)})
            purchase_targets.append(np.log1p(float(df["purchase"].iloc[target_index])))
            redeem_targets.append(np.log1p(float(df["redeem"].iloc[target_index])))
            target_dates.append(target_date)
    return (
        pd.DataFrame(records),
        np.asarray(purchase_targets),
        np.asarray(redeem_targets),
        pd.DatetimeIndex(target_dates),
    )


def build_forecast_matrix(history: pd.DataFrame, forecast_dates) -> pd.DataFrame:
    origin = len(history) - 1
    shared = _history_features(history, origin)
    return pd.DataFrame([
        {**shared, **_forecast_date_features(pd.Timestamp(date), horizon)}
        for horizon, date in enumerate(pd.DatetimeIndex(forecast_dates), start=1)
    ])


def _build_model(seed: int = 42, n_estimators: int = 3000):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.01,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def fit_direct_model(X, y, target_dates, seed: int = 42):
    """Tune on the latest historical month, then refit on all known targets."""
    tune_start = target_dates.max().to_period("M").start_time
    tune_mask = target_dates >= tune_start
    fit_mask = ~tune_mask
    model = _build_model(seed)
    try:
        import lightgbm as lgb

        model.fit(
            X.loc[fit_mask], y[fit_mask],
            eval_set=[(X.loc[tune_mask], y[tune_mask])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        best_iteration = int(getattr(model, "best_iteration_", 0) or 3000)
    except Exception:
        best_iteration = 3000

    final_model = _build_model(seed, n_estimators=best_iteration)
    age_days = (target_dates.max() - target_dates).days.to_numpy(dtype=float)
    sample_weight = np.exp(-np.log(2.0) * age_days / 120.0)
    final_model.fit(X, y, sample_weight=sample_weight)
    return final_model, best_iteration


def fit_and_forecast(history: pd.DataFrame, forecast_dates):
    X, y_purchase, y_redeem, target_dates = build_training_matrix(history)
    purchase_model, purchase_iterations = fit_direct_model(X, y_purchase, target_dates, seed=42)
    redeem_model, redeem_iterations = fit_direct_model(X, y_redeem, target_dates, seed=142)
    X_future = build_forecast_matrix(history, forecast_dates)
    return (
        np.maximum(0.0, np.expm1(purchase_model.predict(X_future))),
        np.maximum(0.0, np.expm1(redeem_model.predict(X_future))),
        purchase_iterations,
        redeem_iterations,
    )


def run_backtest(df: pd.DataFrame, months):
    prediction_frames = []
    metric_rows = []
    for month in months:
        period = pd.Period(month, freq="M")
        history = df[df["date"] < period.start_time].copy()
        actual = df[df["date"].dt.to_period("M") == period].copy()
        p_pred, r_pred, p_iter, r_iter = fit_and_forecast(history, actual["date"])
        result = pd.DataFrame({
            "date": actual["date"].to_numpy(),
            "purchase_true": actual["purchase"].to_numpy(),
            "purchase_pred": p_pred,
            "redeem_true": actual["redeem"].to_numpy(),
            "redeem_pred": r_pred,
            "fold": str(period),
        })
        metrics = evaluate_prediction(result)
        metric_rows.append({"fold": str(period), **metrics, "purchase_iterations": p_iter, "redeem_iterations": r_iter})
        prediction_frames.append(result)
        print(f"[{period}] direct score={metrics['total_score']:.4f}")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    overall = evaluate_prediction(predictions)
    metric_rows.append({"fold": "overall", **overall, "purchase_iterations": 0, "redeem_iterations": 0})
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(DIRECT_BACKTEST_PRED_PATH, index=False, encoding="utf-8-sig")
    metrics.to_csv(DIRECT_BACKTEST_METRICS_PATH, index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False))
    return predictions, metrics


def write_candidate(df: pd.DataFrame):
    forecast_dates = pd.date_range("2014-09-01", "2014-09-30", freq="D")
    p_pred, r_pred, _, _ = fit_and_forecast(df, forecast_dates)
    candidate = pd.DataFrame({
        "report_date": forecast_dates.strftime("%Y%m%d").astype(int),
        "purchase": np.rint(p_pred).astype(int),
        "redeem": np.rint(r_pred).astype(int),
    })
    candidate.to_csv(DIRECT_CANDIDATE_PATH, index=False, header=False, encoding="utf-8")
    print(f"直接多步候选文件：{DIRECT_CANDIDATE_PATH}")
    return candidate


def blend_residual_candidate(baseline: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    """Apply a conservative geometric blend so the proven baseline remains dominant."""
    if not baseline["report_date"].equals(direct["report_date"]):
        raise ValueError("基线与直接多步候选的日期不一致")
    blended = pd.DataFrame({"report_date": baseline["report_date"]})
    for target in ("purchase", "redeem"):
        params = RESIDUAL_BLEND_PARAMS[target]
        weight = params["direct_weight"]
        values = params["scale"] * np.exp(
            (1.0 - weight) * np.log(np.maximum(baseline[target].to_numpy(float), 1.0))
            + weight * np.log(np.maximum(direct[target].to_numpy(float), 1.0))
        )
        blended[target] = np.rint(values).astype(int)
    return blended


def write_residual_candidate(direct: pd.DataFrame):
    baseline = pd.read_csv(
        SUBMISSION_PATH,
        header=None,
        names=["report_date", "purchase", "redeem"],
    )
    blended = blend_residual_candidate(baseline, direct)
    blended.to_csv(RESIDUAL_CANDIDATE_PATH, index=False, header=False, encoding="utf-8")
    print(f"保守残差候选文件：{RESIDUAL_CANDIDATE_PATH}")
    return blended


def redistribute_shape(
    baseline_values,
    direct_values,
    alpha: float,
    max_daily_change: float = SHAPE_MAX_DAILY_CHANGE,
) -> np.ndarray:
    """Borrow only the direct model's daily shape while preserving baseline total.

    The log ratio is centred using baseline values as weights.  After clipping,
    a final normalization restores the exact floating-point monthly total.
    """
    baseline = np.asarray(baseline_values, dtype=float)
    direct = np.asarray(direct_values, dtype=float)
    if baseline.shape != direct.shape or baseline.ndim != 1:
        raise ValueError("基线与直接预测必须是一维且长度相同")
    if np.any(baseline <= 0) or np.any(direct <= 0):
        raise ValueError("日分布调整要求所有预测值大于0")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha 必须位于 [0, 1]")
    if not 0 <= max_daily_change < 1:
        raise ValueError("max_daily_change 必须位于 [0, 1)")

    log_ratio = np.log(direct / baseline)
    centred = log_ratio - np.average(log_ratio, weights=baseline)
    lower = 1.0 - max_daily_change
    upper = 1.0 + max_daily_change

    # Find a common log shift after clipping so the baseline-weighted factor is
    # exactly one. This preserves both the monthly total and the daily bound.
    low_shift, high_shift = -2.0, 2.0
    for _ in range(80):
        shift = (low_shift + high_shift) / 2.0
        factors = np.clip(np.exp(alpha * centred + shift), lower, upper)
        weighted_mean = np.average(factors, weights=baseline)
        if weighted_mean < 1.0:
            low_shift = shift
        else:
            high_shift = shift
    factors = np.clip(
        np.exp(alpha * centred + (low_shift + high_shift) / 2.0),
        lower,
        upper,
    )
    return baseline * factors


def _score_column(y_true, y_pred) -> float:
    error = np.abs(np.asarray(y_pred) - np.asarray(y_true)) / np.maximum(
        np.asarray(y_true, dtype=float), 1.0
    )
    return float(np.maximum(0.0, 10.0 * (1.0 - error / 0.3)).mean())


def tune_shape_alphas(
    baseline_predictions: pd.DataFrame,
    direct_predictions: pd.DataFrame,
):
    """Select target-specific shape strengths using conservative fold stability."""
    direct = direct_predictions.copy()
    direct_folds = set(direct["fold"].astype(str))
    base = baseline_predictions[
        (baseline_predictions["strategy"] == "month_end_boost")
        & baseline_predictions["fold"].astype(str).isin(direct_folds)
    ].copy()
    base["date"] = pd.to_datetime(base["date"])
    direct["date"] = pd.to_datetime(direct["date"])
    merged = base.merge(
        direct[["date", "fold", "purchase_pred", "redeem_pred"]],
        on=["date", "fold"],
        suffixes=("_base", "_direct"),
        validate="one_to_one",
    )
    if len(merged) != len(base):
        raise ValueError("基线回测与直接多步回测的日期不完整匹配")

    rows = []
    selected = {}
    for target in ("purchase", "redeem"):
        fold_names = sorted(merged["fold"].unique())
        base_fold_scores = {
            fold: _score_column(
                group[f"{target}_true"], group[f"{target}_pred_base"]
            )
            for fold, group in merged.groupby("fold")
        }
        candidates = []
        for alpha in SHAPE_ALPHA_GRID:
            adjusted_parts = []
            fold_deltas = []
            for fold, group in merged.groupby("fold", sort=True):
                adjusted = redistribute_shape(
                    group[f"{target}_pred_base"],
                    group[f"{target}_pred_direct"],
                    alpha,
                )
                adjusted_parts.append(pd.Series(adjusted, index=group.index))
                fold_deltas.append(
                    _score_column(group[f"{target}_true"], adjusted)
                    - base_fold_scores[fold]
                )
            adjusted_all = pd.concat(adjusted_parts).sort_index()
            overall = _score_column(merged[f"{target}_true"], adjusted_all)
            base_overall = _score_column(
                merged[f"{target}_true"], merged[f"{target}_pred_base"]
            )
            row = {
                "target": target,
                "alpha": alpha,
                "score": overall,
                "delta": overall - base_overall,
                "winning_folds": sum(delta >= 0 for delta in fold_deltas),
                "worst_fold_delta": min(fold_deltas),
            }
            row.update({f"delta_{fold}": delta for fold, delta in zip(fold_names, fold_deltas)})
            rows.append(row)
            candidates.append(row)

        # Require four of five folds to be non-worse and tightly bound any loss.
        safe = [
            row for row in candidates
            if row["alpha"] > 0
            and row["delta"] > 0
            and row["winning_folds"] >= max(1, len(fold_names) - 1)
            and row["worst_fold_delta"] >= -0.05
        ]
        selected[target] = max(safe, key=lambda row: row["delta"])["alpha"] if safe else 0.0

    return selected, pd.DataFrame(rows)


def build_weekday_shape_signal(
    history_data: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Generate a leakage-free weekday-only shape signal for every backtest fold."""
    frames = []
    baseline = baseline_predictions[
        baseline_predictions["strategy"] == "month_end_boost"
    ]
    for fold, group in baseline.groupby("fold", sort=True):
        period = pd.Period(fold, freq="M")
        history = history_data[history_data["date"] < period.start_time]
        frames.append(pd.DataFrame({
            "date": group["date"].to_numpy(),
            "fold": fold,
            "purchase_pred": weekday_seasonal_forecast(
                history, group["date"], "purchase"
            ),
            "redeem_pred": weekday_seasonal_forecast(
                history, group["date"], "redeem"
            ),
        }))
    return pd.concat(frames, ignore_index=True)


def _round_preserving_total(
    values,
    required_total: int,
    lower_bounds=None,
    upper_bounds=None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    rounded = np.rint(values).astype(np.int64)
    if lower_bounds is not None:
        rounded = np.maximum(rounded, np.asarray(lower_bounds, dtype=np.int64))
    if upper_bounds is not None:
        rounded = np.minimum(rounded, np.asarray(upper_bounds, dtype=np.int64))
    remainder = int(required_total - rounded.sum())
    while remainder:
        residual = values - rounded
        if remainder > 0:
            eligible = np.ones(len(rounded), dtype=bool) if upper_bounds is None else rounded < upper_bounds
            order = np.where(eligible)[0][np.argsort(-residual[eligible])]
            step = 1
        else:
            eligible = np.ones(len(rounded), dtype=bool) if lower_bounds is None else rounded > lower_bounds
            order = np.where(eligible)[0][np.argsort(residual[eligible])]
            step = -1
        if not len(order):
            raise ValueError("日变化边界与月度总量约束无法同时满足")
        count = min(abs(remainder), len(order))
        rounded[order[:count]] += step
        remainder -= step * count
    if int(rounded.sum()) != int(required_total):
        raise AssertionError("整数化后未能保持月度总量")
    return rounded


def write_shape_candidate(shape_signal: pd.DataFrame, selected_alphas: dict[str, float]):
    baseline = pd.read_csv(
        SUBMISSION_PATH,
        header=None,
        names=["report_date", "purchase", "redeem"],
    )
    if not baseline["report_date"].equals(shape_signal["report_date"]):
        raise ValueError("基线与日分布信号的日期不一致")
    candidate = pd.DataFrame({"report_date": baseline["report_date"]})
    for target in ("purchase", "redeem"):
        adjusted = redistribute_shape(
            baseline[target], shape_signal[target], selected_alphas[target]
        )
        candidate[target] = _round_preserving_total(
            adjusted,
            int(baseline[target].sum()),
            lower_bounds=np.ceil(
                baseline[target].to_numpy(float) * (1.0 - SHAPE_MAX_DAILY_CHANGE)
            ).astype(np.int64),
            upper_bounds=np.floor(
                baseline[target].to_numpy(float) * (1.0 + SHAPE_MAX_DAILY_CHANGE)
            ).astype(np.int64),
        )
    candidate.to_csv(SHAPE_CANDIDATE_PATH, index=False, header=False, encoding="utf-8")
    print(f"月总量守恒候选文件：{SHAPE_CANDIDATE_PATH}")
    print(f"日分布调整强度：{selected_alphas}")
    return candidate


def main():
    parser = argparse.ArgumentParser(description="直接多步预测与滚动回测")
    parser.add_argument(
        "--months", nargs="+",
        default=["2014-04", "2014-05", "2014-06", "2014-07", "2014-08"],
    )
    parser.add_argument("--write-candidate", action="store_true")
    args = parser.parse_args()
    df = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    direct_backtest, _ = run_backtest(df, args.months)
    if args.write_candidate:
        direct = write_candidate(df)
        write_residual_candidate(direct)
        baseline_backtest = pd.read_csv(BACKTEST_PRED_PATH, parse_dates=["date"])
        direct_alphas, direct_metrics = tune_shape_alphas(
            baseline_backtest, direct_backtest
        )
        direct_metrics.insert(0, "signal", "direct")
        weekday_backtest = build_weekday_shape_signal(df, baseline_backtest)
        selected_alphas, weekday_metrics = tune_shape_alphas(
            baseline_backtest, weekday_backtest
        )
        weekday_metrics.insert(0, "signal", "weekday")
        pd.concat([direct_metrics, weekday_metrics], ignore_index=True).to_csv(
            SHAPE_BACKTEST_METRICS_PATH, index=False, encoding="utf-8-sig"
        )
        print(f"直接多步信号：{direct_alphas}")
        print(f"星期季节信号：{selected_alphas}")

        forecast_dates = pd.date_range("2014-09-01", "2014-09-30", freq="D")
        weekday_signal = pd.DataFrame({
            "report_date": forecast_dates.strftime("%Y%m%d").astype(int),
            "purchase": weekday_seasonal_forecast(
                df, forecast_dates, "purchase"
            ),
            "redeem": weekday_seasonal_forecast(df, forecast_dates, "redeem"),
        })
        write_shape_candidate(weekday_signal, selected_alphas)


if __name__ == "__main__":
    main()
