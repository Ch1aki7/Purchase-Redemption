"""结构化模型的扩展窗口滚动回测与九月候选文件生成。"""

import argparse

import pandas as pd

from .config import (
    DAILY_FEATURES_PATH,
    STRUCTURAL_BACKTEST_METRICS_PATH,
    STRUCTURAL_BACKTEST_PRED_PATH,
    STRUCTURAL_SUBMISSION_PATH,
    STRUCTURAL_SUBMISSION_WITH_HEADER_PATH,
    STRUCTURAL_VALIDATION_PATH,
)
from .evaluate import evaluate_prediction, relative_error
from .structural_model import StructuralFlowForecaster, StructuralParams
from .backtest import STAGE_MONTHS


def evaluate_mode(data, periods, mode, params):
    predictions = []
    metrics = []
    for period in periods:
        start = period.start_time
        history = data[data["date"] < start]
        actual = data[data["date"].dt.to_period("M") == period]
        model = StructuralFlowForecaster(mode, params).fit(history)
        result = model.predict(actual["date"])
        result = result.merge(
            actual[["date", "purchase", "redeem"]].rename(
                columns={"purchase": "purchase_true", "redeem": "redeem_true"}
            ), on="date", how="inner",
        )
        result["fold"] = str(period)
        result["strategy"] = mode
        result["purchase_error"] = relative_error(result["purchase_true"], result["purchase_pred"])
        result["redeem_error"] = relative_error(result["redeem_true"], result["redeem_pred"])
        score = evaluate_prediction(result)
        metrics.append({"fold": str(period), "strategy": mode, **score})
        predictions.append(result)
        print(f"[{period}] {mode}: score={score['total_score']:.4f}")
    combined = pd.concat(predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(metrics)
    worst = fold_metrics.loc[fold_metrics["risk_adjusted_loss"].idxmax()]
    metrics.append({
        "fold": "overall", "strategy": mode, **evaluate_prediction(combined),
        "worst_fold": str(worst["fold"]),
        "worst_fold_risk_adjusted_loss": float(worst["risk_adjusted_loss"]),
        "fold_risk_adjusted_loss_std": float(fold_metrics["risk_adjusted_loss"].std(ddof=0)),
        "worst_fold_flow_mape": float(fold_metrics["flow_mape"].max()),
        "fold_flow_mape_std": float(fold_metrics["flow_mape"].std(ddof=0)),
    })
    return combined, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="结构化非递归滚动回测")
    parser.add_argument(
        "--stage", choices=list(STAGE_MONTHS), default="all",
        help="development用于参数筛选；confirm仅评估8月；all用于冻结方案后的完整报告",
    )
    parser.add_argument("--months", nargs="+", help="自定义月份；提供后覆盖 --stage")
    parser.add_argument("--modes", nargs="+", choices=StructuralFlowForecaster.MODES, default=list(StructuralFlowForecaster.MODES))
    parser.add_argument("--trend-window", type=int, default=120)
    parser.add_argument("--level-window", type=int, default=28)
    parser.add_argument("--trend-strength", type=float, default=0.5)
    parser.add_argument("--weekday-shrinkage", type=float, default=8.0)
    parser.add_argument("--month-shrinkage", type=float, default=15.0)
    parser.add_argument("--holiday-shrinkage", type=float, default=8.0)
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    data = data[data["date"] >= pd.Timestamp("2013-08-01")].copy()
    month_values = args.months or STAGE_MONTHS[args.stage]
    evaluation_stage = "custom" if args.months else args.stage
    periods = [pd.Period(value, freq="M") for value in month_values]
    params = StructuralParams(
        trend_window=args.trend_window,
        level_window=args.level_window,
        trend_strength=args.trend_strength,
        weekday_shrinkage=args.weekday_shrinkage,
        month_shrinkage=args.month_shrinkage,
        holiday_shrinkage=args.holiday_shrinkage,
    )
    prediction_frames, metric_rows = [], []
    for mode in args.modes:
        predictions, metrics = evaluate_mode(data, periods, mode, params)
        prediction_frames.append(predictions)
        for row in metrics:
            row["evaluation_stage"] = evaluation_stage
        metric_rows.extend(metrics)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_metrics = pd.DataFrame(metric_rows)
    all_predictions.to_csv(STRUCTURAL_BACKTEST_PRED_PATH, index=False, encoding="utf-8-sig")
    all_metrics.to_csv(STRUCTURAL_BACKTEST_METRICS_PATH, index=False, encoding="utf-8-sig")

    overall = all_metrics[all_metrics["fold"] == "overall"].sort_values("risk_adjusted_loss")
    best_mode = overall.iloc[0]["strategy"]
    august = all_predictions[(all_predictions["fold"] == "2014-08") & (all_predictions["strategy"] == best_mode)]
    if not august.empty:
        august.to_csv(STRUCTURAL_VALIDATION_PATH, index=False, encoding="utf-8-sig")

    if evaluation_stage == "all":
        forecast_dates = pd.date_range("2014-09-01", "2014-09-30", freq="D")
        candidate = StructuralFlowForecaster(best_mode, params).fit(data).predict(forecast_dates)
        submission = candidate.assign(
            report_date=candidate["date"].dt.strftime("%Y%m%d").astype(int),
            purchase=candidate["purchase_pred"].round().clip(lower=0).astype("int64"),
            redeem=candidate["redeem_pred"].round().clip(lower=0).astype("int64"),
        )[["report_date", "purchase", "redeem"]]
        submission.to_csv(STRUCTURAL_SUBMISSION_WITH_HEADER_PATH, index=False, encoding="utf-8-sig")
        submission.to_csv(STRUCTURAL_SUBMISSION_PATH, index=False, header=False)
    print("\n结构模型回测汇总：")
    print(all_metrics[[
        "fold", "strategy", "flow_mape", "tail_p90",
        "severe_30_ratio", "risk_adjusted_loss", "total_score",
    ]].to_string(index=False))
    print(f"\n最佳模式：{best_mode}")
    if evaluation_stage == "all":
        print(f"候选提交文件：{STRUCTURAL_SUBMISSION_PATH}")
    else:
        print("当前为开发/确认阶段，不生成或覆盖9月候选提交文件。")


if __name__ == "__main__":
    main()
