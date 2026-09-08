"""Run the complete production-model selection chain on the current data.

The search and forecasting functions are migrated into ``formal_model_core``.
This module owns the current data/path adapters and orchestration, while the
original search spaces, development folds and August gate stay intact.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .formal_model_core import backtest as c3_backtest
from .formal_model_core import calendar_forecast as c3_calendar
from .formal_model_core import event_distance_forecast as c3_event
from .formal_model_core import monthly_shape_forecast as c3_monthly
from .formal_model_core import preprocess as c3_preprocess
from .formal_model_core import train as c3_train

from .config import (
    C3_BACKTEST_METRICS_PATH,
    C3_BACKTEST_PRED_PATH,
    C3_CALENDAR_CANDIDATE_PATH,
    C3_CALENDAR_METRICS_PATH,
    C3_EVENT_DISTANCE_METRICS_PATH,
    C3_EVENT_DISTANCE_VALIDATION_PATH,
    C3_MONTHLY_SHAPE_CANDIDATE_PATH,
    C3_MONTHLY_SHAPE_METRICS_PATH,
    EVENT_DISTANCE_CANDIDATE_PATH,
    EVENT_DISTANCE_MANIFEST_PATH,
    EVENT_DISTANCE_PURCHASE_ONLY_PATH,
    EVENT_DISTANCE_REDEEM_ONLY_PATH,
    PREDICTION_NO_HEADER_PATH,
    PROCESSED_DATA_DIR,
)
from .event_distance import create_target_isolated_submissions, load_submission, normalize_history


FOLDS = tuple(c3_calendar.TUNE_FOLDS) + (c3_calendar.HOLDOUT_FOLD,)


def _notify(callback, message: str):
    if callback is not None:
        callback(message)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def build_c3_features(history: pd.DataFrame) -> pd.DataFrame:
    """Create the production time, lag, holiday and history feature families."""
    data = normalize_history(history)
    data = c3_preprocess.add_time_features(data)
    data = c3_preprocess.add_month_start_end_features(data)
    data = c3_preprocess.add_holiday_features(data)
    data = c3_preprocess.add_lag_rolling_features(data)
    data = c3_preprocess.add_same_period_features(data)
    data = c3_preprocess.add_ratio_diff_features(data)
    numeric = data.select_dtypes(include=[np.number]).columns
    data[numeric] = data[numeric].ffill().bfill().fillna(0)
    return data.sort_values("date")


def _single_thread_rf(seed=42):
    """Production RandomForest parameters with deterministic single-thread execution."""
    return RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=3,
        random_state=seed, n_jobs=1,
    )


def build_c3_backtest(data: pd.DataFrame, callback=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the production five-fold month-end reference backtest."""
    # C3 uses RF when LightGBM is unavailable. Restricting worker count changes
    # neither its parameters nor deterministic predictions and avoids Windows UI
    # subprocess failures.
    original_train_rf = c3_train.build_rf
    original_backtest_rf = c3_backtest.build_rf
    c3_train.build_rf = _single_thread_rf
    c3_backtest.build_rf = _single_thread_rf
    try:
        frame = data[data["date"] >= pd.Timestamp("2013-08-01")].copy()
        feature_cols = c3_train.get_feature_columns(frame)
        prediction_frames = []
        metric_rows = []
        for month in FOLDS:
            _notify(callback, f"正式模型滚动回测：{month}")
            predictions, metrics = c3_backtest.evaluate_fold(
                frame, feature_cols, pd.Period(month, freq="M"), n_seeds=1,
            )
            prediction_frames.extend(predictions)
            metric_rows.extend(metrics)
    finally:
        c3_train.build_rf = original_train_rf
        c3_backtest.build_rf = original_backtest_rf
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(C3_BACKTEST_PRED_PATH, index=False, encoding="utf-8-sig")
    metrics.to_csv(C3_BACKTEST_METRICS_PATH, index=False, encoding="utf-8-sig")
    return predictions, metrics


def run_calendar_stage(data, baseline, submission, callback=None):
    _notify(callback, "日历 Ridge：4—7 月全参数搜索，8 月独立门禁")
    holdout_rows = []
    result_frames = []
    for target in ("purchase", "redeem"):
        params, results = c3_calendar.search_target(data, baseline, target)
        result_frames.append(results)
        _, test = c3_calendar._fold(data, baseline, c3_calendar.HOLDOUT_FOLD)
        prediction = c3_calendar._forecast_from_params(
            data, test["date"], test[f"{target}_pred"], target, params,
        )
        params = {**params}
        params["holdout_delta"] = (
            c3_calendar._score(test[f"{target}_true"], prediction)
            - c3_calendar._score(test[f"{target}_true"], test[f"{target}_pred"])
        )
        params["accepted"] = bool(params["holdout_delta"] > 0)
        holdout_rows.append(params)
    pd.concat([*result_frames, pd.DataFrame(holdout_rows)], ignore_index=True).to_csv(
        C3_CALENDAR_METRICS_PATH, index=False, encoding="utf-8-sig",
    )
    candidate = submission.copy()
    dates = pd.to_datetime(candidate["report_date"].astype(str), format="%Y%m%d")
    for params in holdout_rows:
        target = params["target"]
        if params["accepted"]:
            candidate[target] = np.maximum(1, np.rint(c3_calendar._forecast_from_params(
                data, dates, submission[target], target, params,
            ))).astype(np.int64)
    candidate.to_csv(C3_CALENDAR_CANDIDATE_PATH, index=False, header=False, encoding="utf-8")
    return candidate, holdout_rows


def run_monthly_stage(data, baseline, calendar_candidate, calendar_rows, callback=None):
    _notify(callback, "月度总量/日形状：组件筛选、开发期组合与 8 月门禁")
    reference_params = {row["target"]: row for row in calendar_rows}
    selections = {}
    result_frames = []
    for target in ("purchase", "redeem"):
        selected, results, folds = c3_monthly.select_target(
            data, baseline, target, reference_params[target],
        )
        results.insert(0, "row_type", "grid")
        result_frames.append(results)
        if selected is None:
            selections[target] = {"target": target, "accepted": False}
            continue
        history, test, reference = folds[c3_monthly.HOLDOUT_FOLD]
        prediction = c3_monthly.predict_selected(
            history, test["date"], target, reference, selected,
        )
        selected["holdout_delta"] = (
            c3_calendar._score(test[f"{target}_true"], prediction)
            - c3_calendar._score(test[f"{target}_true"], reference)
        )
        selected["accepted"] = bool(selected["holdout_delta"] > 0)
        selections[target] = selected
    summary = pd.DataFrame(selections.values())
    summary.insert(0, "row_type", "selected")
    pd.concat([*result_frames, summary], ignore_index=True).to_csv(
        C3_MONTHLY_SHAPE_METRICS_PATH, index=False, encoding="utf-8-sig",
    )
    candidate = calendar_candidate.copy()
    dates = pd.to_datetime(candidate["report_date"].astype(str), format="%Y%m%d")
    history = data[data["date"] < dates.min()]
    for target, selected in selections.items():
        if selected.get("accepted", False):
            candidate[target] = np.maximum(1, np.rint(c3_monthly.predict_selected(
                history, dates, target, calendar_candidate[target], selected,
            ))).astype(np.int64)
    candidate.to_csv(C3_MONTHLY_SHAPE_CANDIDATE_PATH, index=False, header=False, encoding="utf-8")
    return candidate, selections


def run_event_stage(data, baseline, monthly_candidate, calendar_rows, callback=None):
    _notify(callback, "正式模型：Ridge/Holt-Winters 全网格与稳健融合搜索")
    reference_params = {row["target"]: row for row in calendar_rows}
    selections = {}
    frames = []
    for target in ("purchase", "redeem"):
        _notify(callback, f"正式模型参数搜索：{target}")
        ridge_top, ridge_all = c3_event.shortlist(data, target, "ridge")
        hw_top, hw_all = c3_event.shortlist(data, target, "hw")
        selected, grid = c3_event.select_ensemble(
            data, baseline, target, reference_params[target], ridge_top, hw_top,
        )
        frames.extend([ridge_all, hw_all, grid])
        selections[target] = selected
    selected_frame = pd.DataFrame([
        value if value is not None else {"row_type": "selected", "target": target}
        for target, value in selections.items()
    ])
    pd.concat([*frames, selected_frame], ignore_index=True).to_csv(
        C3_EVENT_DISTANCE_METRICS_PATH, index=False, encoding="utf-8-sig",
    )
    candidate = monthly_candidate.copy()
    dates = pd.to_datetime(candidate["report_date"].astype(str), format="%Y%m%d")
    history = data[data["date"] < dates.min()]
    for target, selected in selections.items():
        if selected is None:
            continue
        ridge_signal = c3_event.ridge_forecast(
            history, dates, target, json.loads(selected["ridge_spec"]),
        )
        hw_signal = c3_event.holt_winters_forecast(
            history, dates, target, json.loads(selected["hw_spec"]),
        )
        candidate[target] = np.maximum(1, np.rint(c3_event.log_ensemble(
            monthly_candidate[target], ridge_signal, hw_signal,
            float(selected["ridge_weight"]), float(selected["hw_weight"]),
            dates, float(selected["event_scale"]),
        ))).astype(np.int64)
    candidate.to_csv(EVENT_DISTANCE_CANDIDATE_PATH, index=False, header=False, encoding="utf-8")
    return candidate, selections


def build_event_validation_predictions(data, baseline, calendar_rows, selections):
    """Replay selected production models on development and holdout folds."""
    reference_params = {row["target"]: row for row in calendar_rows}
    fold_frames = []
    for fold in FOLDS:
        merged = None
        for target in ("purchase", "redeem"):
            selected = selections.get(target)
            if selected is None:
                continue
            history, test = c3_calendar._fold(data, baseline, fold)
            reference = c3_calendar._forecast_from_params(
                data, test["date"], test[f"{target}_pred"], target,
                reference_params[target],
            )
            ridge_signal = c3_event.ridge_forecast(
                history, test["date"], target, json.loads(selected["ridge_spec"]),
            )
            hw_signal = c3_event.holt_winters_forecast(
                history, test["date"], target, json.loads(selected["hw_spec"]),
            )
            prediction = c3_event.log_ensemble(
                reference, ridge_signal, hw_signal,
                float(selected["ridge_weight"]), float(selected["hw_weight"]),
                test["date"], float(selected["event_scale"]),
            )
            target_frame = pd.DataFrame({
                "date": pd.to_datetime(test["date"]).to_numpy(),
                f"actual_{target}": test[f"{target}_true"].to_numpy(float),
                f"pred_{target}": prediction,
            })
            merged = target_frame if merged is None else merged.merge(target_frame, on="date")
        if merged is not None:
            merged["fold"] = fold
            merged["role"] = "holdout" if fold == c3_calendar.HOLDOUT_FOLD else "development"
            fold_frames.append(merged)
    validation = pd.concat(fold_frames, ignore_index=True).sort_values("date")
    validation.to_csv(C3_EVENT_DISTANCE_VALIDATION_PATH, index=False, encoding="utf-8-sig")
    return validation


def run_full_pipeline(history: pd.DataFrame | None = None, callback=None) -> dict:
    """Execute the complete production-model chain and create both isolated submissions."""
    if history is None:
        history = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    submission = load_submission(PREDICTION_NO_HEADER_PATH)
    data = build_c3_features(history)
    predictions, backtest_metrics = build_c3_backtest(data, callback)
    baseline = predictions[predictions["strategy"] == "month_end_boost"].copy()
    calendar_candidate, calendar_rows = run_calendar_stage(data, baseline, submission, callback)
    monthly_candidate, monthly_selections = run_monthly_stage(
        data, baseline, calendar_candidate, calendar_rows, callback,
    )
    candidate, event_selections = run_event_stage(
        data, baseline, monthly_candidate, calendar_rows, callback,
    )
    _notify(callback, "回放已选正式模型，生成实际值与预测值明细")
    validation_predictions = build_event_validation_predictions(
        data, baseline, calendar_rows, event_selections,
    )
    purchase_only, redeem_only = create_target_isolated_submissions(
        C3_MONTHLY_SHAPE_CANDIDATE_PATH, EVENT_DISTANCE_CANDIDATE_PATH,
    )
    manifest = _jsonable({
        "pipeline": ["正式模型滚动回测", "日历预测", "月度总量与日形状预测", "正式模型预测", "目标隔离导出"],
        "source_submission": str(PREDICTION_NO_HEADER_PATH),
        "development_folds": list(c3_calendar.TUNE_FOLDS),
        "holdout_fold": c3_calendar.HOLDOUT_FOLD,
        "calendar_selected": calendar_rows,
        "monthly_shape_selected": monthly_selections,
        "event_distance_selected": event_selections,
        "outputs": {
            "purchase_only": str(EVENT_DISTANCE_PURCHASE_ONLY_PATH),
            "redeem_only": str(EVENT_DISTANCE_REDEEM_ONLY_PATH),
            "validation_predictions": str(C3_EVENT_DISTANCE_VALIDATION_PATH),
        },
    })
    EVENT_DISTANCE_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    _notify(callback, "完整链路结束，目标隔离文件已生成")
    return {
        "candidate": candidate, "purchase_only": purchase_only,
        "redeem_only": redeem_only, "manifest": manifest,
        "backtest_metrics": backtest_metrics,
        "validation_predictions": validation_predictions,
    }


if __name__ == "__main__":
    result = run_full_pipeline(callback=print)
    print(json.dumps(result["manifest"]["outputs"], ensure_ascii=False, indent=2))
