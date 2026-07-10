import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from .config import (
    DAILY_FEATURES_PATH,
    VALIDATION_PRED_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_PURCHASE_PATH,
    MODEL_REDEEM_PATH,
)
from .evaluate import evaluate_prediction


def build_model():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    except Exception:
        print("未能使用 LightGBM，自动降级为 RandomForestRegressor。")
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )


def get_feature_columns(df: pd.DataFrame):
    exclude = {"date", "purchase", "redeem"}
    cols = [c for c in df.columns if c not in exclude]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def main():
    if not DAILY_FEATURES_PATH.exists():
        raise FileNotFoundError("请先运行 python -m src.preprocess 或 python run_all.py")

    df = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values("date")

    # 前 30 天 lag 特征不完整，去掉更稳妥
    df = df[df["date"] >= pd.Timestamp("2013-07-31")].copy()

    train_df = df[df["date"] <= pd.Timestamp("2014-07-31")].copy()
    valid_df = df[(df["date"] >= pd.Timestamp("2014-08-01")) & (df["date"] <= pd.Timestamp("2014-08-31"))].copy()

    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("训练集或验证集为空，请检查数据日期范围是否覆盖 2013-07-01 至 2014-08-31。")

    feature_cols = get_feature_columns(df)
    X_train = train_df[feature_cols].fillna(0)
    X_valid = valid_df[feature_cols].fillna(0)

    y_purchase_train = np.log1p(train_df["purchase"])
    y_redeem_train = np.log1p(train_df["redeem"])

    model_purchase = build_model()
    model_redeem = build_model()

    model_purchase.fit(X_train, y_purchase_train)
    model_redeem.fit(X_train, y_redeem_train)

    purchase_pred = np.expm1(model_purchase.predict(X_valid))
    redeem_pred = np.expm1(model_redeem.predict(X_valid))
    purchase_pred = np.maximum(0, purchase_pred)
    redeem_pred = np.maximum(0, redeem_pred)

    valid_result = pd.DataFrame({
        "date": valid_df["date"],
        "purchase_true": valid_df["purchase"].values,
        "purchase_pred": purchase_pred,
        "redeem_true": valid_df["redeem"].values,
        "redeem_pred": redeem_pred,
    })
    valid_result.to_csv(VALIDATION_PRED_PATH, index=False, encoding="utf-8-sig")

    metrics = evaluate_prediction(valid_result)
    print("验证集评估结果（模拟评分）：")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("MAE purchase:", mean_absolute_error(valid_df["purchase"], purchase_pred))
    print("MAE redeem:", mean_absolute_error(valid_df["redeem"], redeem_pred))
    print("MAPE purchase:", mean_absolute_percentage_error(valid_df["purchase"], purchase_pred))
    print("MAPE redeem:", mean_absolute_percentage_error(valid_df["redeem"], redeem_pred))

    joblib.dump(model_purchase, MODEL_PURCHASE_PATH)
    joblib.dump(model_redeem, MODEL_REDEEM_PATH)
    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print(f"模型已保存到：{MODEL_PURCHASE_PATH} 和 {MODEL_REDEEM_PATH}")
    print(f"验证集预测已保存到：{VALIDATION_PRED_PATH}")


if __name__ == "__main__":
    main()
