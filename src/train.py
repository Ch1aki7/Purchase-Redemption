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


def build_lightgbm():
    """LightGBM 调参版：更多树 + 早停 + 正则化（调参实验最优配置）。"""
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.003,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=5,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def build_rf():
    """RandomForest 作为集成成员 + 降级方案。"""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def build_model():
    try:
        from lightgbm import LGBMRegressor
        return build_lightgbm()
    except Exception:
        print("未能使用 LightGBM，自动降级为 RandomForestRegressor。")
        return build_rf()


def get_feature_columns(df: pd.DataFrame):
    exclude = {"date", "purchase", "redeem"}
    cols = [c for c in df.columns if c not in exclude]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def fit_single_model(model, X_train, y_train, X_valid=None, y_valid=None, use_early_stop=False):
    """训练单个模型，可选早停（仅 LightGBM）。"""
    if use_early_stop and X_valid is not None and y_valid is not None:
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[
                    __import__("lightgbm").early_stopping(stopping_rounds=100, verbose=False),
                    __import__("lightgbm").log_evaluation(period=0),
                ],
            )
            return model
        except Exception:
            pass
    model.fit(X_train, y_train)
    return model


class EnsembleModel:
    """LightGBM + RandomForest 加权集成（含目标变换的预测还原能力）。"""

    def __init__(self, lgb_weight=0.7, rf_weight=0.3):
        self.lgb_weight = lgb_weight
        self.rf_weight = rf_weight
        self.lgb_model = None
        self.rf_model = None
        self.use_ensemble = True

    def fit(self, X_train, y_train, X_valid, y_valid):
        # LightGBM 用早停
        self.lgb_model = build_lightgbm()
        self.lgb_model = fit_single_model(
            self.lgb_model, X_train, y_train, X_valid, y_valid, use_early_stop=True
        )
        # RandomForest 无早停
        try:
            self.rf_model = build_rf()
            self.rf_model.fit(X_train, y_train)
        except Exception as e:
            print(f"RandomForest 训练失败，仅用 LightGBM：{e}")
            self.use_ensemble = False
        return self

    def predict(self, X):
        lgb_pred = self.lgb_model.predict(X)
        if not self.use_ensemble or self.rf_model is None:
            return lgb_pred
        rf_pred = self.rf_model.predict(X)
        return self.lgb_weight * lgb_pred + self.rf_weight * rf_pred


def main():
    if not DAILY_FEATURES_PATH.exists():
        raise FileNotFoundError("请先运行 python -m src.preprocess 或 python run_all.py")

    df = pd.read_csv(DAILY_FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values("date")

    # 前 30 天 lag 特征不完整，去掉更稳妥
    df = df[df["date"] >= pd.Timestamp("2013-08-01")].copy()

    train_df = df[df["date"] <= pd.Timestamp("2014-07-31")].copy()
    valid_df = df[(df["date"] >= pd.Timestamp("2014-08-01")) & (df["date"] <= pd.Timestamp("2014-08-31"))].copy()

    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("训练集或验证集为空，请检查数据日期范围是否覆盖 2013-07-01 至 2014-08-31。")

    feature_cols = get_feature_columns(df)
    X_train = train_df[feature_cols].fillna(0)
    X_valid = valid_df[feature_cols].fillna(0)

    y_purchase_train = np.log1p(train_df["purchase"])
    y_redeem_train = np.log1p(train_df["redeem"])
    y_purchase_valid = np.log1p(valid_df["purchase"])
    y_redeem_valid = np.log1p(valid_df["redeem"])

    # 训练集成模型
    print("训练申购集成模型...")
    model_purchase = EnsembleModel(lgb_weight=0.95, rf_weight=0.05)
    model_purchase.fit(X_train, y_purchase_train, X_valid, y_purchase_valid)

    print("训练赎回集成模型...")
    model_redeem = EnsembleModel(lgb_weight=0.95, rf_weight=0.05)
    model_redeem.fit(X_train, y_redeem_train, X_valid, y_redeem_valid)

    # 验证集预测
    purchase_pred = np.expm1(model_purchase.predict(X_valid))
    redeem_pred = np.expm1(model_redeem.predict(X_valid))

    # 裁剪到历史合理区间（训练集 P1~P99）
    for tgt, pred_arr in [("purchase", purchase_pred), ("redeem", redeem_pred)]:
        lo = np.percentile(train_df[tgt], 1)
        hi = np.percentile(train_df[tgt], 99)
        pred_arr[:] = np.clip(pred_arr, lo, hi)
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

    # 保存集成模型对象
    joblib.dump(model_purchase, MODEL_PURCHASE_PATH)
    joblib.dump(model_redeem, MODEL_REDEEM_PATH)
    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print(f"模型已保存到：{MODEL_PURCHASE_PATH} 和 {MODEL_REDEEM_PATH}")
    print(f"验证集预测已保存到：{VALIDATION_PRED_PATH}")


if __name__ == "__main__":
    main()
