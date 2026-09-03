import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    DAILY_FEATURES_PATH,
    VALIDATION_PRED_PATH,
    SUBMISSION_WITH_HEADER_PATH,
    ROOT_DIR,
    DATA_DIR,
)
from src.evaluate import evaluate_prediction, mock_score_from_error, relative_error

st.set_page_config(page_title="资金流入流出预测系统", layout="wide")
st.title("资金流入流出预测系统")
st.caption("数据探索、模型训练与评估、预测结果展示、误差分析")


# =========================
# 通用工具函数
# =========================

def read_csv_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def find_data_file(keyword: str):
    if not DATA_DIR.exists():
        return None
    candidates = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.txt"))
    for p in candidates:
        if keyword.lower() in p.name.lower():
            return p
    return None


def read_raw_table(keyword: str):
    path = find_data_file(keyword)
    if path is None:
        return None
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None


def get_feature_columns(df: pd.DataFrame):
    exclude = {"date", "purchase", "redeem"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def classify_anomaly(date):
    date = pd.Timestamp(date)
    if date.day == 1:
        return "月初异常"
    if date.day >= date.days_in_month - 2:
        return "月末异常"
    if date.dayofweek == 6:
        return "周日波动"
    if date.strftime("%m-%d") == "08-08":
        return "孤立高值"
    if date.day >= date.days_in_month - 4:
        return "月末附近"
    return "普通日期"


def enrich_prediction_result(valid: pd.DataFrame) -> pd.DataFrame:
    valid = valid.copy()
    valid["date"] = pd.to_datetime(valid["date"])
    valid["purchase_abs_error"] = (valid["purchase_pred"] - valid["purchase_true"]).abs()
    valid["redeem_abs_error"] = (valid["redeem_pred"] - valid["redeem_true"]).abs()
    valid["purchase_error"] = relative_error(valid["purchase_true"], valid["purchase_pred"])
    valid["redeem_error"] = relative_error(valid["redeem_true"], valid["redeem_pred"])
    valid["purchase_score"] = mock_score_from_error(valid["purchase_error"])
    valid["redeem_score"] = mock_score_from_error(valid["redeem_error"])
    valid["daily_weighted_score"] = valid["purchase_score"] * 0.45 + valid["redeem_score"] * 0.55
    valid["purchase_direction"] = np.where(valid["purchase_pred"] > valid["purchase_true"], "高估", "低估")
    valid["redeem_direction"] = np.where(valid["redeem_pred"] > valid["redeem_true"], "高估", "低估")
    valid["is_zero_score_day"] = (valid["purchase_error"] > 0.3) | (valid["redeem_error"] > 0.3)
    valid["异常类型"] = valid["date"].apply(classify_anomaly)
    return valid


def build_model(model_name: str, n_estimators: int, max_depth: int, learning_rate: float, random_state: int):
    if model_name == "RandomForest":
        depth = None if max_depth == 0 else max_depth
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=depth,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "GradientBoosting":
        depth = 3 if max_depth == 0 else max_depth
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=depth,
            random_state=random_state,
        )
    if model_name == "LightGBM":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise RuntimeError("当前环境未安装 LightGBM，请选择 RandomForest 或 GradientBoosting。") from exc
        depth = -1 if max_depth == 0 else max_depth
        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=depth,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"未知模型：{model_name}")


def run_cv(model_name, X, y, n_estimators, max_depth, learning_rate, random_state):
    rows = []
    tscv = TimeSeriesSplit(n_splits=3)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        model = build_model(model_name, n_estimators, max_depth, learning_rate, random_state + fold)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = np.expm1(model.predict(X.iloc[val_idx]))
        true = np.expm1(y.iloc[val_idx])
        rows.append({"折数": fold, "MAPE": mean_absolute_percentage_error(true, pred)})
    return pd.DataFrame(rows)


def build_loss_curve(model_name, X_train, y_train, X_valid, y_valid, n_estimators, max_depth, learning_rate, random_state):
    points = sorted(set([max(5, int(n_estimators * r)) for r in [0.2, 0.4, 0.6, 0.8, 1.0]]))
    rows = []
    for n in points:
        model = build_model(model_name, n, max_depth, learning_rate, random_state)
        model.fit(X_train, y_train)
        train_pred = np.expm1(model.predict(X_train))
        valid_pred = np.expm1(model.predict(X_valid))
        train_true = np.expm1(y_train)
        valid_true = np.expm1(y_valid)
        rows.append({
            "训练轮数/树数": n,
            "训练MAPE": mean_absolute_percentage_error(train_true, train_pred),
            "验证MAPE": mean_absolute_percentage_error(valid_true, valid_pred),
        })
    return pd.DataFrame(rows)


def train_interactive_model(model_name, n_estimators, max_depth, learning_rate, random_state, use_log_target=True):
    df = read_csv_if_exists(DAILY_FEATURES_PATH)
    if df is None:
        raise FileNotFoundError("缺少 output/daily_features.csv，请先运行完整流程生成特征文件。")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df[df["date"] >= pd.Timestamp("2013-08-01")].copy()
    train_df = df[df["date"] <= pd.Timestamp("2014-07-31")].copy()
    valid_df = df[(df["date"] >= pd.Timestamp("2014-08-01")) & (df["date"] <= pd.Timestamp("2014-08-31"))].copy()
    feature_cols = get_feature_columns(df)
    X_train = train_df[feature_cols].fillna(0)
    X_valid = valid_df[feature_cols].fillna(0)

    if use_log_target:
        y_purchase_train = np.log1p(train_df["purchase"])
        y_redeem_train = np.log1p(train_df["redeem"])
    else:
        y_purchase_train = train_df["purchase"].astype(float)
        y_redeem_train = train_df["redeem"].astype(float)

    p_model = build_model(model_name, n_estimators, max_depth, learning_rate, random_state)
    r_model = build_model(model_name, n_estimators, max_depth, learning_rate, random_state + 100)
    p_model.fit(X_train, y_purchase_train)
    r_model.fit(X_train, y_redeem_train)

    if use_log_target:
        purchase_pred = np.expm1(p_model.predict(X_valid))
        redeem_pred = np.expm1(r_model.predict(X_valid))
    else:
        purchase_pred = p_model.predict(X_valid)
        redeem_pred = r_model.predict(X_valid)

    purchase_pred = np.maximum(0, purchase_pred)
    redeem_pred = np.maximum(0, redeem_pred)
    result = pd.DataFrame({
        "date": valid_df["date"].values,
        "purchase_true": valid_df["purchase"].values,
        "purchase_pred": purchase_pred,
        "redeem_true": valid_df["redeem"].values,
        "redeem_pred": redeem_pred,
    })
    result = enrich_prediction_result(result)

    cv_purchase = run_cv(model_name, X_train, y_purchase_train, n_estimators, max_depth, learning_rate, random_state)
    cv_redeem = run_cv(model_name, X_train, y_redeem_train, n_estimators, max_depth, learning_rate, random_state + 100)
    cv_df = cv_purchase.rename(columns={"MAPE": "申购MAPE"}).merge(
        cv_redeem.rename(columns={"MAPE": "赎回MAPE"}), on="折数"
    )
    cv_df["加权MAPE"] = cv_df["申购MAPE"] * 0.45 + cv_df["赎回MAPE"] * 0.55

    loss_purchase = build_loss_curve(model_name, X_train, y_purchase_train, X_valid, np.log1p(valid_df["purchase"]), n_estimators, max_depth, learning_rate, random_state)
    loss_purchase["目标"] = "申购"
    loss_redeem = build_loss_curve(model_name, X_train, y_redeem_train, X_valid, np.log1p(valid_df["redeem"]), n_estimators, max_depth, learning_rate, random_state + 100)
    loss_redeem["目标"] = "赎回"
    loss_df = pd.concat([loss_purchase, loss_redeem], ignore_index=True)

    return result, cv_df, loss_df, feature_cols


def get_display_validation(source_name: str):
    if source_name == "本次交互训练结果" and "interactive_valid" in st.session_state:
        return st.session_state["interactive_valid"].copy()
    valid = read_csv_if_exists(VALIDATION_PRED_PATH)
    if valid is None:
        return None
    return enrich_prediction_result(valid)


# =========================
# 侧边栏：运行与结果选择
# =========================
with st.sidebar:
    st.divider()
    st.header("展示结果来源")
    source_options = ["主流程滚动预测结果"]
    if "interactive_valid" in st.session_state:
        source_options.append("本次交互训练结果")
    result_source = st.radio("选择当前展示结果", source_options)


# =========================
# 页面主体
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["数据探索", "模型训练与评估", "预测结果展示", "误差分析"])

with tab1:
    st.subheader("数据探索模块")
    df = read_csv_if_exists(DAILY_FEATURES_PATH)
    if df is None:
        st.warning("尚未生成 daily_features.csv，请先运行：python run_all.py")
    else:
        df["date"] = pd.to_datetime(df["date"])
        c1, c2, c3 = st.columns(3)
        c1.metric("日级样本数", len(df))
        c2.metric("特征列数", len(df.columns))
        c3.metric("日期范围", f"{df['date'].min().date()} ~ {df['date'].max().date()}")
        st.dataframe(df.head(20), use_container_width=True)
        fund_plot = df[["date", "purchase", "redeem"]].melt(id_vars="date", var_name="类型", value_name="金额")
        st.plotly_chart(px.line(fund_plot, x="date", y="金额", color="类型", title="历史申购/赎回趋势图"), use_container_width=True)

        yield_cols = [c for c in ["mfd_daily_yield", "mfd_7daily_yield"] if c in df.columns]
        if yield_cols:
            yield_plot = df[["date"] + yield_cols].melt("date", var_name="收益率字段", value_name="数值")
            st.plotly_chart(px.line(yield_plot, x="date", y="数值", color="收益率字段", title="收益率变化图"), use_container_width=True)

        shibor_cols = [c for c in df.columns if c.startswith("Interest_")]
        if shibor_cols:
            chosen_shibor = st.multiselect("选择要展示的 Shibor 指标", shibor_cols, default=shibor_cols[: min(3, len(shibor_cols))])
            if chosen_shibor:
                shibor_plot = df[["date"] + chosen_shibor].melt("date", var_name="Shibor字段", value_name="利率")
                st.plotly_chart(px.line(shibor_plot, x="date", y="利率", color="Shibor字段", title="Shibor 利率变化图"), use_container_width=True)

    profile = read_raw_table("user_profile")
    if profile is None:
        st.info("未找到用户画像原始表，用户分布图需要解压原始数据后展示。")
    else:
        st.markdown("### 用户分布")
        col1, col2, col3 = st.columns(3)
        if "sex" in profile.columns:
            sex_counts = profile["sex"].fillna("未知").astype(str).value_counts().reset_index()
            sex_counts.columns = ["性别", "人数"]
            col1.plotly_chart(px.pie(sex_counts, names="性别", values="人数", title="性别分布"), use_container_width=True)
        if "city" in profile.columns:
            city_counts = profile["city"].fillna("未知").astype(str).value_counts().head(10).reset_index()
            city_counts.columns = ["城市", "人数"]
            col2.plotly_chart(px.bar(city_counts, x="城市", y="人数", title="城市Top10"), use_container_width=True)
        if "constellation" in profile.columns:
            cons_counts = profile["constellation"].fillna("未知").astype(str).value_counts().reset_index()
            cons_counts.columns = ["星座", "人数"]
            col3.plotly_chart(px.bar(cons_counts, x="星座", y="人数", title="星座分布"), use_container_width=True)

with tab2:
    st.subheader("模型训练与评估模块")
    st.markdown("### 选择模型和参数进行真实训练")
    st.info("本模块会在页面中真实训练申购模型和赎回模型，并把训练结果保存到当前 Streamlit 会话，用于后续预测对比和误差分析展示。")

    col1, col2, col3, col4 = st.columns(4)
    model_name = col1.selectbox("模型", ["RandomForest", "GradientBoosting", "LightGBM"])
    n_estimators = col2.slider("树数量/训练轮数", min_value=20, max_value=500, value=100, step=20)
    max_depth = col3.slider("最大深度（0表示不限制/默认）", min_value=0, max_value=20, value=6, step=1)
    learning_rate = col4.select_slider("学习率", options=[0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1], value=0.03)
    random_state = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1)

    if st.button("训练并更新展示结果", type="primary"):
        try:
            with st.spinner("正在训练模型、生成损失曲线和交叉验证结果..."):
                result, cv_df, loss_df, feature_cols = train_interactive_model(
                    model_name,
                    int(n_estimators),
                    int(max_depth),
                    float(learning_rate),
                    int(random_state),
                    use_log_target=True,
                )
            st.session_state["interactive_valid"] = result
            st.session_state["interactive_cv"] = cv_df
            st.session_state["interactive_loss"] = loss_df
            st.session_state["interactive_meta"] = {
                "model": model_name,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "feature_count": len(feature_cols),
            }
            st.success("训练完成，已更新当前会话中的交互训练结果。可在侧边栏切换展示来源。")
        except Exception as exc:
            st.error(f"训练失败：{exc}")

    display_valid = get_display_validation(result_source)
    if display_valid is None:
        st.warning("暂无验证集结果，请先运行完整流程或在本页训练模型。")
    else:
        metrics = evaluate_prediction(display_valid)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("申购 MAPE", f"{metrics['purchase_mape']:.4f}")
        c2.metric("赎回 MAPE", f"{metrics['redeem_mape']:.4f}")
        c3.metric("模拟总分", f"{metrics['total_score']:.4f}")
        c4.metric("零分日", f"{int(display_valid['is_zero_score_day'].sum())} 天")

        purchase_plot = display_valid[["date", "purchase_true", "purchase_pred"]].melt("date", var_name="类型", value_name="金额")
        redeem_plot = display_valid[["date", "redeem_true", "redeem_pred"]].melt("date", var_name="类型", value_name="金额")
        st.plotly_chart(px.line(purchase_plot, x="date", y="金额", color="类型", title=f"申购：真实值 vs 预测值（{result_source}）"), use_container_width=True)
        st.plotly_chart(px.line(redeem_plot, x="date", y="金额", color="类型", title=f"赎回：真实值 vs 预测值（{result_source}）"), use_container_width=True)

    if "interactive_loss" in st.session_state:
        st.markdown("### 训练过程损失曲线")
        loss_df = st.session_state["interactive_loss"]
        loss_plot = loss_df.melt(["训练轮数/树数", "目标"], value_vars=["训练MAPE", "验证MAPE"], var_name="曲线", value_name="MAPE")
        st.plotly_chart(px.line(loss_plot, x="训练轮数/树数", y="MAPE", color="曲线", line_dash="目标", markers=True, title="训练过程损失曲线"), use_container_width=True)

    if "interactive_cv" in st.session_state:
        st.markdown("### 时间序列交叉验证结果")
        st.dataframe(st.session_state["interactive_cv"], use_container_width=True)
        cv_plot = st.session_state["interactive_cv"].melt("折数", value_vars=["申购MAPE", "赎回MAPE", "加权MAPE"], var_name="指标", value_name="MAPE")
        st.plotly_chart(px.bar(cv_plot, x="折数", y="MAPE", color="指标", barmode="group", title="交叉验证 MAPE"), use_container_width=True)

with tab3:
    st.subheader("预测结果展示模块")
    pred = read_csv_if_exists(SUBMISSION_WITH_HEADER_PATH)
    if pred is None:
        st.warning("尚未生成预测结果文件，请先运行：python run_all.py")
    else:
        pred["date"] = pd.to_datetime(pred["report_date"].astype(str))
        st.markdown("### 未来30天预测结果（2014年9月）")
        st.dataframe(pred[["report_date", "purchase", "redeem"]], use_container_width=True)
        pred_plot = pred[["date", "purchase", "redeem"]].melt("date", var_name="类型", value_name="金额")
        st.plotly_chart(px.line(pred_plot, x="date", y="金额", color="类型", title="2014年9月申购/赎回预测"), use_container_width=True)
        submit_bytes = pred[["report_date", "purchase", "redeem"]].to_csv(index=False, header=False).encode("utf-8-sig")
        st.download_button("下载天池提交文件（无表头）", submit_bytes, "tc_comp_predict_table.csv")

    display_valid = get_display_validation(result_source)
    if display_valid is not None:
        st.markdown("### 验证集真实值对比与每日误差/得分")
        show_cols = [
            "date", "purchase_true", "purchase_pred", "purchase_abs_error", "purchase_error", "purchase_score",
            "redeem_true", "redeem_pred", "redeem_abs_error", "redeem_error", "redeem_score", "daily_weighted_score",
        ]
        st.dataframe(display_valid[show_cols], use_container_width=True)

with tab4:
    st.subheader("误差分析模块")
    display_valid = get_display_validation(result_source)
    if display_valid is None:
        st.warning("暂无验证集结果，请先运行完整流程或在模型训练页训练模型。")
    else:
        show_cols = [
            "date", "异常类型",
            "purchase_true", "purchase_pred", "purchase_abs_error", "purchase_error", "purchase_score", "purchase_direction",
            "redeem_true", "redeem_pred", "redeem_abs_error", "redeem_error", "redeem_score", "redeem_direction",
            "daily_weighted_score",
        ]
        st.markdown("### 每日误差、得分与高估/低估方向")
        st.dataframe(
            display_valid[show_cols].style.background_gradient(
                subset=["purchase_error", "redeem_error", "purchase_abs_error", "redeem_abs_error"], cmap="Reds"
            ),
            use_container_width=True,
        )

        st.markdown("### 零分日高亮（相对误差 > 30%）")
        zero_days = display_valid[display_valid["is_zero_score_day"]].copy()
        if zero_days.empty:
            st.success("当前展示结果没有误差超过30%的零分日。")
        else:
            st.dataframe(zero_days[show_cols], use_container_width=True)
            zero_plot = zero_days[["date", "purchase_error", "redeem_error"]].melt("date", var_name="误差类型", value_name="相对误差")
            st.plotly_chart(px.bar(zero_plot, x="date", y="相对误差", color="误差类型", barmode="group", title="零分日相对误差"), use_container_width=True)

        st.markdown("### 改进方向提示")
        st.info(
            "若零分日集中在月初/月末，说明需要更强的月内周期或后处理策略；"
            "若集中在周日，说明周末资金行为波动大；若是孤立高值，则可能需要外部业务信息辅助判断。"
        )
