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
from .predict import build_future_row


def build_lightgbm(seed=42, n_estimators=10000):
    """LightGBM 调参版：更多树 + 小学习率 + 早停 + 正则化（第四轮调优实验最优配置）。

    第四轮调优实验发现：learning_rate=0.001, n_estimators=10000 比原配置
    (0.003, 5000) 略优，且配合多种子集成可进一步降低方差。
    """
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.001,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=5,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def build_rf(seed=42):
    """RandomForest 作为集成成员 + 降级方案。"""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )


def build_model(seed=42):
    try:
        return build_lightgbm(seed=seed)
    except Exception as exc:
        print(f"未能使用 LightGBM（{exc}），自动降级为 RandomForestRegressor。")
        return build_rf(seed=seed)


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
    """多种子 LightGBM 集成（第四轮调优发现纯LightGBM多种子平均最优）。

    内部训练 n_seeds 个不同随机种子的 LightGBM，预测时取平均。
    这比单模型方差更小，比加入 RandomForest 效果更好（RF 在该数据上贡献为负）。
    含目标变换的预测还原能力。
    """

    def __init__(self, lgb_weight=1.0, rf_weight=0.0, n_seeds=3):
        self.lgb_weight = lgb_weight
        self.rf_weight = rf_weight
        self.n_seeds = n_seeds
        self.lgb_models = []
        self.rf_model = None
        self.use_ensemble = False  # 是否使用 RF

    def fit(self, X_train, y_train):
        """在训练期内部早停选轮数，再用完整训练期重训最终模型。"""
        tune_size = min(31, max(1, len(X_train) // 5))
        can_early_stop = len(X_train) > tune_size
        if can_early_stop:
            X_fit, X_tune = X_train.iloc[:-tune_size], X_train.iloc[-tune_size:]
            y_fit, y_tune = y_train.iloc[:-tune_size], y_train.iloc[-tune_size:]

        # 多种子模型；LightGBM 不可用或训练失败时自动降级为 RandomForest。
        self.lgb_models = []
        for seed in range(42, 42 + self.n_seeds):
            m = build_model(seed=seed)
            is_lightgbm = m.__class__.__module__.startswith("lightgbm")
            try:
                if is_lightgbm and can_early_stop:
                    tuned = fit_single_model(
                        m, X_fit, y_fit, X_tune, y_tune, use_early_stop=True
                    )
                    best_iteration = int(getattr(tuned, "best_iteration_", 0) or 0)
                    final_estimators = best_iteration if best_iteration > 0 else 10000
                    m = build_lightgbm(seed=seed, n_estimators=final_estimators)
                m.fit(X_train, y_train)
            except Exception as exc:
                if not is_lightgbm:
                    raise
                print(
                    f"LightGBM 训练失败（{exc}），"
                    f"种子 {seed} 自动降级为 RandomForestRegressor。"
                )
                m = build_rf(seed=seed)
                m.fit(X_train, y_train)
            self.lgb_models.append(m)
        # RandomForest 保留接口但不使用（调优实验证明权重0最优）
        if self.rf_weight > 0:
            try:
                self.rf_model = build_rf()
                self.rf_model.fit(X_train, y_train)
                self.use_ensemble = True
            except Exception as e:
                print(f"RandomForest 训练失败：{e}")
        return self

    def predict(self, X):
        # 多种子 LightGBM 平均
        preds = [m.predict(X) for m in self.lgb_models]
        lgb_pred = np.mean(preds, axis=0)
        if not self.use_ensemble or self.rf_model is None or self.rf_weight == 0:
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
    # 训练集成模型：只在训练期内部早停，8 月完全留作最终验证。
    print("训练申购集成模型...")
    model_purchase = EnsembleModel(lgb_weight=1.0, rf_weight=0.0)
    model_purchase.fit(X_train, y_purchase_train)

    print("训练赎回集成模型...")
    model_redeem = EnsembleModel(lgb_weight=1.0, rf_weight=0.0)
    model_redeem.fit(X_train, y_redeem_train)

    # 验证集采用滚动预测（与9月预测逻辑一致，避免使用8月真实历史导致分数虚高）
    # 逐天预测8月1日→8月31日，预测值回填为下一天的 lag 特征
    print("验证集滚动预测（模拟真实预测场景）...")
    # 构造8月外部变量：用7月均值（8月真实值在预测时未知，用过去信息）
    non_exog = {"date", "purchase", "redeem"}
    lag_roll_prefixes = ("purchase_lag_", "redeem_lag_", "purchase_roll_", "redeem_roll_",
                         "purchase_same_", "redeem_same_", "purchase_diff_", "redeem_diff_",
                         "net_inflow", "purchase_redeem_ratio")
    time_cols = {"dayofweek", "dayofmonth", "month", "is_weekend", "is_month_start", "is_month_end",
                 "dayofweek_sin", "dayofweek_cos", "dayofmonth_sin", "dayofmonth_cos",
                 "month_sin", "month_cos", "days_to_month_start", "days_to_month_end",
                 "is_first_3_days", "is_last_3_days", "is_first_7_days", "is_last_7_days",
                 "month_start_weight", "month_end_weight",
                 "purchase_month_start_mean", "redeem_month_start_mean", "is_holiday",
                 "days_to_next_holiday", "days_from_last_holiday",
                 "is_day_before_holiday", "is_day_after_holiday"}
    exog_cols = [
        c for c in feature_cols
        if c not in non_exog and c not in time_cols and not c.startswith(lag_roll_prefixes)
    ]
    july_df = df[(df["date"] >= pd.Timestamp("2014-07-01")) & (df["date"] <= pd.Timestamp("2014-07-31"))]
    exog_values_aug = {}
    for c in exog_cols:
        if c in july_df.columns:
            exog_values_aug[c] = float(july_df[c].mean())

    # 历史裁剪区间（基于训练集）
    p_lo = np.percentile(train_df["purchase"], 1)
    p_hi = np.percentile(train_df["purchase"], 99)
    r_lo = np.percentile(train_df["redeem"], 1)
    r_hi = np.percentile(train_df["redeem"], 99)

    # 滚动预测8月
    history = train_df[["date", "purchase", "redeem"]].copy()
    valid_dates = pd.date_range("2014-08-01", "2014-08-31", freq="D")
    pred_rows = []
    for date in valid_dates:
        X = build_future_row(date, history, feature_cols, exog_values_aug, df)
        p_pred = float(np.expm1(model_purchase.predict(X)[0]))
        r_pred = float(np.expm1(model_redeem.predict(X)[0]))
        # 平滑修正：模型预测 × 0.99 + 近7日均值 × 0.01，缓解滚动漂移
        # 第四轮调优实验：smooth_w=0.99 比 0.95 略优（5.19 vs 5.15）
        recent7_p = float(history["purchase"].tail(7).mean())
        recent7_r = float(history["redeem"].tail(7).mean())
        p_pred = 0.99 * p_pred + 0.01 * recent7_p
        r_pred = 0.99 * r_pred + 0.01 * recent7_r
        if date.day >= date.days_in_month - 2:
            p_pred *= 1.2
            r_pred *= 1.3
        p_pred = float(np.clip(p_pred, p_lo, p_hi))
        r_pred = float(np.clip(r_pred, r_lo, r_hi))
        p_pred = max(0, p_pred)
        r_pred = max(0, r_pred)
        pred_rows.append({"date": date, "purchase_pred": p_pred, "redeem_pred": r_pred})
        # 预测值回填为下一天的 lag 特征（关键：与9月预测一致）
        history = pd.concat([
            history,
            pd.DataFrame({"date": [date], "purchase": [p_pred], "redeem": [r_pred]})
        ], ignore_index=True)

    pred_df = pd.DataFrame(pred_rows)
    # 对齐真实值
    true_df = valid_df[["date", "purchase", "redeem"]].rename(
        columns={"purchase": "purchase_true", "redeem": "redeem_true"})
    valid_result = pred_df.merge(true_df, on="date")
    valid_result = valid_result[["date", "purchase_true", "purchase_pred", "redeem_true", "redeem_pred"]]
    valid_result.to_csv(VALIDATION_PRED_PATH, index=False, encoding="utf-8-sig")

    metrics = evaluate_prediction(valid_result)
    print("验证集评估结果（线性近似公式模拟评分）：")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("MAE purchase:", mean_absolute_error(valid_result["purchase_true"], valid_result["purchase_pred"]))
    print("MAE redeem:", mean_absolute_error(valid_result["redeem_true"], valid_result["redeem_pred"]))
    print("MAPE purchase:", mean_absolute_percentage_error(valid_result["purchase_true"], valid_result["purchase_pred"]))
    print("MAPE redeem:", mean_absolute_percentage_error(valid_result["redeem_true"], valid_result["redeem_pred"]))

    # 8 月只用于上面的无泄露评估。评估完成后，用截至 8 月 31 日的全部
    # 已知数据重训正式模型，供 9 月预测使用。
    print("使用截至 2014-08-31 的全部数据重训正式申购模型...")
    X_final = df[feature_cols].fillna(0)
    y_purchase_final = np.log1p(df["purchase"])
    y_redeem_final = np.log1p(df["redeem"])

    final_model_purchase = EnsembleModel(lgb_weight=1.0, rf_weight=0.0)
    final_model_purchase.fit(X_final, y_purchase_final)
    print("使用截至 2014-08-31 的全部数据重训正式赎回模型...")
    final_model_redeem = EnsembleModel(lgb_weight=1.0, rf_weight=0.0)
    final_model_redeem.fit(X_final, y_redeem_final)

    joblib.dump(final_model_purchase, MODEL_PURCHASE_PATH)
    joblib.dump(final_model_redeem, MODEL_REDEEM_PATH)
    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print(f"模型已保存到：{MODEL_PURCHASE_PATH} 和 {MODEL_REDEEM_PATH}")
    print(f"验证集预测已保存到：{VALIDATION_PRED_PATH}")


if __name__ == "__main__":
    main()
