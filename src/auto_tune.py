"""Resumable random search using leakage-safe expanding-window backtests.

The expensive model fit is shared by many smoothing and seasonal-blend settings,
so one model configuration can produce dozens of recorded parameter trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import (
    build_exogenous_values,
    recursive_forecast,
    train_refitted_ensemble,
)
from .config import DAILY_FEATURES_PATH, OUTPUT_DIR
from .evaluate import evaluate_prediction
from .predict import weekday_anchor
from .train import get_feature_columns


PROFILES = {
    "quick": {
        "trials": 8,
        "months": ["2014-07", "2014-08"],
        "smooth_weights": [0.97, 0.99, 1.0],
        "purchase_weights": [0.95, 1.0],
        "redeem_weights": [0.35, 0.5, 0.65, 0.8],
    },
    "large": {
        "trials": 30,
        "months": ["2014-04", "2014-05", "2014-06", "2014-07", "2014-08"],
        "smooth_weights": [0.95, 0.97, 0.99, 1.0],
        "purchase_weights": [0.9, 0.95, 1.0],
        "redeem_weights": [0.25, 0.35, 0.5, 0.65, 0.8, 1.0],
    },
}


XGBOOST_BASELINE = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 2,
    "min_child_weight": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
}


LIGHTGBM_BASELINE = {
    "n_estimators": 3000,
    "learning_rate": 0.005,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 12,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
}

MODEL_PARAMETER_NAMES = {
    "xgboost": tuple(XGBOOST_BASELINE),
    "lightgbm": tuple(LIGHTGBM_BASELINE),
}


def stable_id(payload: dict, prefix: str) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{prefix}_{hashlib.sha1(encoded).hexdigest()[:12]}"


def parse_number_list(value: str | None, fallback: list[float]) -> list[float]:
    if value is None:
        return fallback
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise ValueError("参数列表不能为空")
    return values


def sample_xgboost(rng: np.random.Generator) -> dict:
    tree_pairs = [(4000, 0.005), (3000, 0.01), (2500, 0.015),
                  (2000, 0.02), (1500, 0.03), (1000, 0.05)]
    n_estimators, learning_rate = tree_pairs[int(rng.integers(len(tree_pairs)))]
    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": int(rng.choice([2, 3, 4, 5])),
        "min_child_weight": int(rng.choice([3, 5, 8, 12, 20])),
        "subsample": float(rng.choice([0.7, 0.85, 1.0])),
        "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 0.05, 0.1, 0.5, 1.0])),
        "reg_lambda": float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0])),
    }


def sample_lightgbm(rng: np.random.Generator) -> dict:
    tree_pairs = [(6000, 0.002), (4000, 0.005), (2500, 0.01),
                  (1500, 0.02), (1000, 0.03)]
    n_estimators, learning_rate = tree_pairs[int(rng.integers(len(tree_pairs)))]
    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": int(rng.choice([7, 15, 31, 63])),
        "max_depth": int(rng.choice([3, 5, 7, -1])),
        "min_child_samples": int(rng.choice([5, 10, 20, 30])),
        "subsample": float(rng.choice([0.7, 0.85, 1.0])),
        "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 0.05, 0.1, 0.5, 1.0])),
        "reg_lambda": float(rng.choice([0.0, 0.1, 0.5, 1.0, 2.0])),
    }


def generate_model_configs(models: list[str], count: int, random_seed: int) -> list[dict]:
    rng = np.random.default_rng(random_seed)
    configs = []
    for model_name in models:
        baseline = XGBOOST_BASELINE if model_name == "xgboost" else LIGHTGBM_BASELINE
        configs.append({"model": model_name, "params": baseline, "source": "baseline"})

    next_model = 0
    seen = {stable_id(config, "model") for config in configs}
    while len(configs) < count:
        model_name = models[next_model % len(models)]
        next_model += 1
        params = sample_xgboost(rng) if model_name == "xgboost" else sample_lightgbm(rng)
        config = {"model": model_name, "params": params, "source": "random"}
        config_id = stable_id(config, "model")
        if config_id not in seen:
            seen.add(config_id)
            configs.append(config)
    return configs[:count]


def load_model_config(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    model_name = data.get("model")
    if model_name not in MODEL_PARAMETER_NAMES:
        raise ValueError(f"参数文件中的 model 必须是 xgboost 或 lightgbm：{path}")
    params = {
        key: data[key]
        for key in MODEL_PARAMETER_NAMES[model_name]
        if key in data and data[key] is not None
    }
    missing = set(MODEL_PARAMETER_NAMES[model_name]) - set(params)
    if missing:
        raise ValueError(f"参数文件缺少模型参数：{sorted(missing)}")
    return {"model": model_name, "params": params, "source": "parameter_file"}, data


def candidate_payload(model_config, smooth_weight, purchase_weight, redeem_weight,
                      months, n_seeds, start_seed):
    return {
        "evaluation_version": 2,
        **model_config,
        "smooth_weight": smooth_weight,
        "purchase_model_weight": purchase_weight,
        "redeem_model_weight": redeem_weight,
        "months": months,
        "n_seeds": n_seeds,
        "start_seed": start_seed,
    }


def atomic_write_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def blend_prediction(base, history, purchase_weight, redeem_weight):
    result = base.copy()
    p_lo, p_hi = np.percentile(history["purchase"], [1, 99])
    r_lo, r_hi = np.percentile(history["redeem"], [1, 99])
    p_anchor = np.array([
        weekday_anchor(date, history, "purchase") for date in result["date"]
    ])
    r_anchor = np.array([
        weekday_anchor(date, history, "redeem") for date in result["date"]
    ])
    result["purchase_pred"] = np.clip(
        purchase_weight * result["purchase_pred"].to_numpy()
        + (1 - purchase_weight) * p_anchor,
        p_lo, p_hi,
    )
    result["redeem_pred"] = np.clip(
        redeem_weight * result["redeem_pred"].to_numpy()
        + (1 - redeem_weight) * r_anchor,
        r_lo, r_hi,
    )
    return result


def evaluate_model_config(
    df, feature_cols, model_config, periods, n_seeds, start_seed,
    smooth_weights, purchase_weights, redeem_weights,
):
    fold_frames = defaultdict(list)
    fold_rows = []
    fit_started = time.perf_counter()

    for period in periods:
        history = df[df["date"] < period.start_time].copy()
        actual = df[df["date"].dt.to_period("M") == period].copy()
        if history.empty or actual.empty:
            raise ValueError(f"回测月份 {period} 缺少历史或真实数据")
        purchase_model, purchase_iterations = train_refitted_ensemble(
            history, feature_cols, "purchase", n_seeds,
            model_name=model_config["model"], preset="default",
            model_params=model_config["params"], start_seed=start_seed,
        )
        redeem_model, redeem_iterations = train_refitted_ensemble(
            history, feature_cols, "redeem", n_seeds,
            model_name=model_config["model"], preset="default",
            model_params=model_config["params"], start_seed=start_seed,
        )
        exogenous = build_exogenous_values(df, feature_cols, period)

        for smooth_weight in smooth_weights:
            base = recursive_forecast(
                purchase_model, redeem_model, history,
                pd.date_range(period.start_time, period.end_time.normalize(), freq="D"),
                feature_cols, exogenous, apply_month_end_boost=False,
                purchase_model_weight=1.0, redeem_model_weight=1.0,
                smooth_weight=smooth_weight,
            )
            for purchase_weight in purchase_weights:
                for redeem_weight in redeem_weights:
                    payload = candidate_payload(
                        model_config, smooth_weight, purchase_weight, redeem_weight,
                        [str(value) for value in periods], n_seeds, start_seed,
                    )
                    trial_id = stable_id(payload, "trial")
                    result = blend_prediction(
                        base, history, purchase_weight, redeem_weight
                    ).merge(
                        actual[["date", "purchase", "redeem"]].rename(
                            columns={"purchase": "purchase_true", "redeem": "redeem_true"}
                        ),
                        on="date",
                    )
                    metrics = evaluate_prediction(result)
                    p_error = np.abs(result["purchase_pred"] - result["purchase_true"]) / result["purchase_true"].replace(0, 1)
                    r_error = np.abs(result["redeem_pred"] - result["redeem_true"]) / result["redeem_true"].replace(0, 1)
                    result["purchase_error"] = p_error
                    result["redeem_error"] = r_error
                    fold_frames[trial_id].append(result)
                    fold_rows.append({
                        "trial_id": trial_id,
                        "model_config_id": stable_id(model_config, "model"),
                        "fold": str(period),
                        **metrics,
                        "purchase_zero_score_days": int((p_error > 0.3).sum()),
                        "redeem_zero_score_days": int((r_error > 0.3).sum()),
                        "purchase_best_iterations": "/".join(map(str, purchase_iterations)),
                        "redeem_best_iterations": "/".join(map(str, redeem_iterations)),
                    })

    elapsed = time.perf_counter() - fit_started
    summary_rows = []
    fold_frame = pd.DataFrame(fold_rows)
    for smooth_weight in smooth_weights:
        for purchase_weight in purchase_weights:
            for redeem_weight in redeem_weights:
                payload = candidate_payload(
                    model_config, smooth_weight, purchase_weight, redeem_weight,
                    [str(value) for value in periods], n_seeds, start_seed,
                )
                trial_id = stable_id(payload, "trial")
                combined = pd.concat(fold_frames[trial_id], ignore_index=True)
                metrics = evaluate_prediction(combined)
                trial_folds = fold_frame[fold_frame["trial_id"] == trial_id]
                summary_rows.append({
                    "trial_id": trial_id,
                    "model_config_id": stable_id(model_config, "model"),
                    "model": model_config["model"],
                    "source": model_config["source"],
                    **model_config["params"],
                    "smooth_weight": smooth_weight,
                    "purchase_model_weight": purchase_weight,
                    "redeem_model_weight": redeem_weight,
                    "months": ",".join(str(value) for value in periods),
                    "n_seeds": n_seeds,
                    "start_seed": start_seed,
                    **metrics,
                    "worst_fold": str(trial_folds.loc[trial_folds["risk_adjusted_loss"].idxmax(), "fold"]),
                    "worst_fold_risk_adjusted_loss": float(trial_folds["risk_adjusted_loss"].max()),
                    "fold_risk_adjusted_loss_std": float(trial_folds["risk_adjusted_loss"].std(ddof=0)),
                    "worst_fold_flow_mape": float(trial_folds["flow_mape"].max()),
                    "fold_flow_mape_std": float(trial_folds["flow_mape"].std(ddof=0)),
                    # 旧字段仅用于兼容既有结果读取。
                    "worst_fold_score": float(trial_folds["total_score"].min()),
                    "fold_score_std": float(trial_folds["total_score"].std(ddof=0)),
                    "purchase_zero_score_days": int((combined["purchase_error"] > 0.3).sum()),
                    "redeem_zero_score_days": int((combined["redeem_error"] > 0.3).sum()),
                    "fit_elapsed_seconds": elapsed,
                })
    return pd.DataFrame(summary_rows), fold_frame


def write_best_json(results: pd.DataFrame, path: Path):
    if results.empty:
        return
    ranked = results.sort_values(
        ["risk_adjusted_loss", "worst_fold_risk_adjusted_loss", "fold_risk_adjusted_loss_std"],
        ascending=[True, True, True],
    )
    best = ranked.iloc[0].where(pd.notna(ranked.iloc[0]), None).to_dict()
    for key, value in list(best.items()):
        if isinstance(value, np.generic):
            best[key] = value.item()
    path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="大量参数随机搜索与递归滚动回测")
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--models", choices=["xgboost", "lightgbm", "both"], default="xgboost")
    parser.add_argument("--trials", type=int, help="模型参数配置数量；包含每种模型的基线")
    parser.add_argument("--months", nargs="+", help="覆盖配置档的回测月份")
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=20260904)
    parser.add_argument("--smooth-weights", help="逗号分隔，例如 0.97,0.99,1.0")
    parser.add_argument("--purchase-weights", help="逗号分隔，例如 0.95,1.0")
    parser.add_argument("--redeem-weights", help="逗号分隔，例如 0.35,0.5,0.65")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "tuning")
    parser.add_argument("--params-json", type=Path, help="只复核一个最佳参数JSON；默认同时复用其中的三个后处理权重")
    parser.add_argument("--no-resume", action="store_true", help="忽略已有结果并从头记录")
    parser.add_argument("--dry-run", action="store_true", help="只展示规模，不训练")
    return parser.parse_args()


def main():
    args = parse_args()
    profile = PROFILES[args.profile]
    trial_count = args.trials or profile["trials"]
    months = args.months or profile["months"]
    parameter_file_data = None
    if args.params_json:
        model_config, parameter_file_data = load_model_config(args.params_json)
        configs = [model_config]
        trial_count = 1
    smooth_fallback = (
        [float(parameter_file_data["smooth_weight"])]
        if parameter_file_data else profile["smooth_weights"]
    )
    purchase_fallback = (
        [float(parameter_file_data["purchase_model_weight"])]
        if parameter_file_data else profile["purchase_weights"]
    )
    redeem_fallback = (
        [float(parameter_file_data["redeem_model_weight"])]
        if parameter_file_data else profile["redeem_weights"]
    )
    smooth_weights = parse_number_list(args.smooth_weights, smooth_fallback)
    purchase_weights = parse_number_list(args.purchase_weights, purchase_fallback)
    redeem_weights = parse_number_list(args.redeem_weights, redeem_fallback)
    if trial_count < 1 or args.n_seeds < 1:
        raise ValueError("--trials 和 --n-seeds 必须大于等于1")
    models = ["xgboost", "lightgbm"] if args.models == "both" else [args.models]
    if not args.params_json and trial_count < len(models):
        raise ValueError("--trials 不能小于待测试模型数量")
    if not args.params_json:
        configs = generate_model_configs(models, trial_count, args.random_seed)
    postprocess_count = len(smooth_weights) * len(purchase_weights) * len(redeem_weights)
    total_candidates = len(configs) * postprocess_count
    total_fits = len(configs) * len(months) * 2 * args.n_seeds
    print(f"配置档={args.profile}，模型参数配置={len(configs)}，后处理组合/模型={postprocess_count}")
    print(f"共记录 {total_candidates} 个候选；预计训练 {total_fits} 个目标模型")
    print(f"回测月份：{', '.join(months)}；输出目录：{args.output_dir}")
    if args.dry_run:
        print(json.dumps(configs[:3], ensure_ascii=False, indent=2))
        return
    if not DAILY_FEATURES_PATH.exists():
        raise FileNotFoundError("缺少 daily_features.csv，请先运行 python -m src.preprocess")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "auto_tune_results.csv"
    folds_path = args.output_dir / "auto_tune_fold_metrics.csv"
    failures_path = args.output_dir / "auto_tune_failures.csv"
    best_path = args.output_dir / "auto_tune_best.json"
    results = pd.DataFrame() if args.no_resume else load_csv(results_path)
    folds = pd.DataFrame() if args.no_resume else load_csv(folds_path)
    failures = pd.DataFrame() if args.no_resume else load_csv(failures_path)
    completed = set(results.get("trial_id", pd.Series(dtype=str)).astype(str))

    df = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] >= pd.Timestamp("2013-08-01")].copy()
    feature_cols = get_feature_columns(df)
    periods = [pd.Period(value, freq="M") for value in months]
    current_trial_ids = {
        stable_id(candidate_payload(
            model_config, smooth, purchase, redeem,
            [str(value) for value in periods], args.n_seeds, args.start_seed,
        ), "trial")
        for model_config in configs
        for smooth in smooth_weights
        for purchase in purchase_weights
        for redeem in redeem_weights
    }

    for index, model_config in enumerate(configs, start=1):
        expected_ids = {
            stable_id(candidate_payload(
                model_config, smooth, purchase, redeem,
                [str(value) for value in periods], args.n_seeds, args.start_seed,
            ), "trial")
            for smooth in smooth_weights
            for purchase in purchase_weights
            for redeem in redeem_weights
        }
        if expected_ids.issubset(completed):
            print(f"[{index}/{len(configs)}] 已完成，跳过 {stable_id(model_config, 'model')}")
            continue
        print(f"[{index}/{len(configs)}] 评估 {model_config['model']} {model_config['params']}")
        try:
            new_results, new_folds = evaluate_model_config(
                df, feature_cols, model_config, periods, args.n_seeds,
                args.start_seed, smooth_weights, purchase_weights, redeem_weights,
            )
            results = pd.concat([results, new_results], ignore_index=True)
            results = results.drop_duplicates("trial_id", keep="last")
            folds = pd.concat([folds, new_folds], ignore_index=True)
            folds = folds.drop_duplicates(["trial_id", "fold"], keep="last")
            atomic_write_csv(results, results_path)
            atomic_write_csv(folds, folds_path)
            current_results = results[
                results["trial_id"].astype(str).isin(current_trial_ids)
            ]
            write_best_json(current_results, best_path)
            completed.update(new_results["trial_id"].astype(str))
            best_loss = current_results["risk_adjusted_loss"].min()
            print(f"  已记录 {len(new_results)} 个组合；当前最低风险损失={best_loss:.6f}")
        except Exception as exc:
            failure = pd.DataFrame([{
                "time": pd.Timestamp.now().isoformat(),
                "model_config_id": stable_id(model_config, "model"),
                "model": model_config["model"],
                "params": json.dumps(model_config["params"], ensure_ascii=False),
                "error": repr(exc),
            }])
            failures = pd.concat([failures, failure], ignore_index=True)
            atomic_write_csv(failures, failures_path)
            print(f"  失败，已记录：{exc}")

    current_results = results[
        results.get("trial_id", pd.Series(dtype=str)).astype(str).isin(current_trial_ids)
    ]
    if current_results.empty:
        raise RuntimeError(f"没有成功的调参结果，请检查 {failures_path}")
    ranked = current_results.sort_values(
        ["risk_adjusted_loss", "worst_fold_risk_adjusted_loss", "fold_risk_adjusted_loss_std"],
        ascending=[True, True, True],
    )
    print("\n风险调整损失最低的前10组：")
    columns = [
        "model", "risk_adjusted_loss", "flow_mape", "tail_p90",
        "severe_30_ratio", "worst_fold", "worst_fold_risk_adjusted_loss",
        "fold_risk_adjusted_loss_std", "total_score", "smooth_weight",
        "purchase_model_weight", "redeem_model_weight", "model_config_id",
    ]
    print(ranked[columns].head(10).to_string(index=False))
    print(f"\n完整排名：{results_path}")
    print(f"逐月明细：{folds_path}")
    print(f"最佳参数：{best_path}")


if __name__ == "__main__":
    main()
