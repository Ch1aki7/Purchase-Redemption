import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from .config import (
    DAILY_FEATURES_PATH,
    VALIDATION_PRED_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_PURCHASE_PATH,
    MODEL_REDEEM_PATH,
)
from .evaluate import evaluate_prediction


def build_model(model_type="lightgbm", n_estimators=400, learning_rate=0.03, max_depth=-1, num_leaves=31):
    model_type = (model_type or "lightgbm").lower()
    if model_type == "randomforest":
        depth = None if max_depth is None or max_depth < 0 else int(max_depth)
        return RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=depth,
            random_state=42,
            n_jobs=-1,
        )

    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            num_leaves=int(num_leaves),
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    except Exception:
        print("未能使用 LightGBM，自动降级为 RandomForestRegressor。")
        return build_model(
            model_type="randomforest",
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth and max_depth > 0 else 12,
        )


def get_feature_columns(df: pd.DataFrame):
    exclude = {"date", "purchase", "redeem"}
    cols = [c for c in df.columns if c not in exclude]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def load_prepared_frame():
    if not DAILY_FEATURES_PATH.exists():
        raise FileNotFoundError("请先运行 python -m src.preprocess 或 python run_all.py")
    df = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    return df[df["date"] >= pd.Timestamp("2013-07-31")].copy()


def split_train_valid(df: pd.DataFrame):
    train_df = df[df["date"] <= pd.Timestamp("2014-07-31")].copy()
    valid_df = df[(df["date"] >= pd.Timestamp("2014-08-01")) & (df["date"] <= pd.Timestamp("2014-08-31"))].copy()
    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("训练集或验证集为空，请检查数据日期范围是否覆盖 2013-07-01 至 2014-08-31。")
    return train_df, valid_df


def _fit_one(model, X_train, y_train, X_valid=None, y_valid=None):
    """训练单个模型；LightGBM 可记录验证集损失曲线。"""
    history = None
    model_name = type(model).__name__.lower()
    if "lgbm" in model_name and X_valid is not None and y_valid is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_names=["train", "valid"],
            eval_metric="l2",
        )
        history = getattr(model, "evals_result_", None)
    else:
        model.fit(X_train, y_train)
    return model, history


def run_time_series_cv(df: pd.DataFrame, feature_cols, model_params, n_splits=3):
    """在训练集上做时序交叉验证，返回各折 MAPE / 模拟总分。"""
    train_df, _ = split_train_valid(df)
    X = train_df[feature_cols].fillna(0).values
    y_purchase = np.log1p(train_df["purchase"].values)
    y_redeem = np.log1p(train_df["redeem"].values)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
        model_p = build_model(**model_params)
        model_r = build_model(**model_params)
        model_p.fit(X[tr_idx], y_purchase[tr_idx])
        model_r.fit(X[tr_idx], y_redeem[tr_idx])

        purchase_pred = np.maximum(0, np.expm1(model_p.predict(X[va_idx])))
        redeem_pred = np.maximum(0, np.expm1(model_r.predict(X[va_idx])))
        fold_df = pd.DataFrame({
            "purchase_true": np.expm1(y_purchase[va_idx]),
            "purchase_pred": purchase_pred,
            "redeem_true": np.expm1(y_redeem[va_idx]),
            "redeem_pred": redeem_pred,
        })
        metrics = evaluate_prediction(fold_df)
        rows.append({
            "fold": fold,
            "train_size": int(len(tr_idx)),
            "valid_size": int(len(va_idx)),
            "purchase_mape": metrics["purchase_mape"],
            "redeem_mape": metrics["redeem_mape"],
            "total_score": metrics["total_score"],
        })
    return pd.DataFrame(rows)


def train_and_evaluate(model_params=None, save=True, run_cv=True, n_splits=3):
    """供命令行与可视化系统复用的训练入口。"""
    model_params = dict(model_params or {})
    model_params.setdefault("model_type", "lightgbm")
    model_params.setdefault("n_estimators", 400)
    model_params.setdefault("learning_rate", 0.03)
    model_params.setdefault("max_depth", -1)
    model_params.setdefault("num_leaves", 31)

    df = load_prepared_frame()
    train_df, valid_df = split_train_valid(df)
    feature_cols = get_feature_columns(df)

    X_train = train_df[feature_cols].fillna(0)
    X_valid = valid_df[feature_cols].fillna(0)
    y_purchase_train = np.log1p(train_df["purchase"])
    y_redeem_train = np.log1p(train_df["redeem"])
    y_purchase_valid = np.log1p(valid_df["purchase"])
    y_redeem_valid = np.log1p(valid_df["redeem"])

    model_purchase = build_model(**model_params)
    model_redeem = build_model(**model_params)
    model_purchase, purchase_history = _fit_one(
        model_purchase, X_train, y_purchase_train, X_valid, y_purchase_valid
    )
    model_redeem, redeem_history = _fit_one(
        model_redeem, X_train, y_redeem_train, X_valid, y_redeem_valid
    )

    purchase_pred = np.maximum(0, np.expm1(model_purchase.predict(X_valid)))
    redeem_pred = np.maximum(0, np.expm1(model_redeem.predict(X_valid)))

    valid_result = pd.DataFrame({
        "date": valid_df["date"].values,
        "purchase_true": valid_df["purchase"].values,
        "purchase_pred": purchase_pred,
        "redeem_true": valid_df["redeem"].values,
        "redeem_pred": redeem_pred,
    })
    metrics = evaluate_prediction(valid_result)
    metrics["mae_purchase"] = float(mean_absolute_error(valid_df["purchase"], purchase_pred))
    metrics["mae_redeem"] = float(mean_absolute_error(valid_df["redeem"], redeem_pred))
    metrics["mape_sklearn_purchase"] = float(
        mean_absolute_percentage_error(valid_df["purchase"], purchase_pred)
    )
    metrics["mape_sklearn_redeem"] = float(
        mean_absolute_percentage_error(valid_df["redeem"], redeem_pred)
    )

    cv_results = run_time_series_cv(df, feature_cols, model_params, n_splits=n_splits) if run_cv else None

    if save:
        valid_result.to_csv(VALIDATION_PRED_PATH, index=False, encoding="utf-8-sig")
        joblib.dump(model_purchase, MODEL_PURCHASE_PATH)
        joblib.dump(model_redeem, MODEL_REDEEM_PATH)
        with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    return {
        "metrics": metrics,
        "valid_result": valid_result,
        "purchase_history": purchase_history,
        "redeem_history": redeem_history,
        "cv_results": cv_results,
        "model_params": model_params,
        "feature_cols": feature_cols,
    }


def main():
    result = train_and_evaluate(save=True, run_cv=False)
    metrics = result["metrics"]
    print("验证集评估结果（模拟评分）：")
    for k in ["purchase_mape", "redeem_mape", "purchase_score", "redeem_score", "total_score"]:
        print(f"  {k}: {metrics[k]:.6f}")
    print("MAE purchase:", metrics["mae_purchase"])
    print("MAE redeem:", metrics["mae_redeem"])
    print("MAPE purchase:", metrics["mape_sklearn_purchase"])
    print("MAPE redeem:", metrics["mape_sklearn_redeem"])
    print(f"模型已保存到：{MODEL_PURCHASE_PATH} 和 {MODEL_REDEEM_PATH}")
    print(f"验证集预测已保存到：{VALIDATION_PRED_PATH}")


if __name__ == "__main__":
    main()
