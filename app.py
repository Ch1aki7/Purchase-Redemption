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
    DATA_QUALITY_REPORT_PATH,
    VALIDATION_PRED_PATH,
    SUBMISSION_WITH_HEADER_PATH,
    ROOT_DIR,
    DATA_DIR,
)
from src.evaluate import evaluate_prediction, mock_score_from_error, relative_error

st.set_page_config(
    page_title="FlowScope · 资金流动预测工作台",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fb; color: #172033; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
        border-right: 1px solid #dbe3ee;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #172033;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #596780;
    }
    [data-testid="stSidebar"] hr { border-color: #dbe3ee; }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #d7e0eb;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary * {
        color: #24324a;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        color: #ffffff;
        background: #0f766e;
        border: 1px solid #0f766e;
        font-weight: 650;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        color: #ffffff;
        background: #115e59;
        border-color: #115e59;
    }
    [data-testid="stSidebar"] [data-testid="stProgress"] > div > div > div {
        background-color: #0f766e;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] p {
        color: inherit;
    }
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px; color: white;
        background: linear-gradient(120deg, #0f172a 0%, #123f62 55%, #0f766e 100%);
        box-shadow: 0 12px 32px rgba(15, 23, 42, .16); margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .45rem 0 0; color: #dbeafe; }
    .section-note { color: #64748b; font-size: .92rem; margin-top: -.55rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: .8rem 1rem; box-shadow: 0 4px 14px rgba(15, 23, 42, .05);
    }
    div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
        background: white; border-radius: 14px; padding: .25rem;
    }
    .status-dot { color: #34d399; font-size: .85rem; }
    </style>
    <div class="hero">
      <h1>FlowScope 资金流动预测工作台</h1>
      <p>从历史资金行为到未来流动性预警 · 申购与赎回一体化分析</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# 通用工具函数
# =========================

def read_csv_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def format_amount(value) -> str:
    """用适合业务阅读的单位展示大额资金。"""
    value = float(value)
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f} 亿"
    if abs(value) >= 1e4:
        return f"{value / 1e4:.1f} 万"
    return f"{value:,.0f}"


def add_net_flow(df: pd.DataFrame, purchase_col="purchase", redeem_col="redeem") -> pd.DataFrame:
    result = df.copy()
    result["net_flow"] = result[purchase_col] - result[redeem_col]
    result["flow_status"] = np.where(result["net_flow"] >= 0, "净流入", "净流出")
    return result


def base_figure(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=55, b=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig


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
    st.markdown("## 💹 FlowScope")
    st.caption("资金预测与流动性决策平台")
    artifact_checks = {
        "特征数据": DAILY_FEATURES_PATH.exists(),
        "验证结果": VALIDATION_PRED_PATH.exists(),
        "未来预测": SUBMISSION_WITH_HEADER_PATH.exists(),
    }
    ready_count = sum(artifact_checks.values())
    st.progress(ready_count / len(artifact_checks), text=f"系统就绪度 {ready_count}/{len(artifact_checks)}")
    for label, ready in artifact_checks.items():
        st.caption(f"{'●' if ready else '○'} {label} · {'已就绪' if ready else '待生成'}")
    st.divider()
    st.header("运行控制")
    st.info("一键运行会重新生成当前唯一的正式组合模型、匹配的8月验证结果和9月提交文件。")
    with st.expander("完整流程固定参数说明", expanded=False):
        st.markdown(
            """
            **正式预测采用约 131 分提交所对应的组合结构。**

            - 训练数据：`2013-08-01` 至 `2014-07-31`
            - 选参方式：4—7月多窗口滚动回测，8月作为独立门禁
            - 申购结构：日历趋势基线 + 月度总量/日内形状分解
            - 赎回结构：上述基线 + 节假日前后距离特征 + 周周期信号
            - 组合策略：保留月度形状申购，仅替换经过线上验证有效的赎回预测
            - 正式预测：`2014-09-01` 至 `2014-09-30`
            - 输出文件：`output/validation_prediction.csv`、`output/tc_comp_predict_table.csv` 和带表头展示文件
            - 说明：内部仍计算树模型滚动基线作为组合输入，但不再把旧模型作为前端结果展示
            """
        )
    if st.button("一键运行完整流程", type="secondary"):
        with st.spinner("正在执行 run_all.py，可能需要几分钟..."):
            result = subprocess.run(
                [sys.executable, "run_all.py"],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        if result.returncode == 0:
            st.success("完整流程运行完成，已刷新正式结果文件。")
        else:
            st.error("完整流程运行失败，请查看输出信息。")
        with st.expander("运行输出"):
            st.code((result.stdout or "") + "\n" + (result.stderr or ""))

    st.divider()
    st.success("当前结果源：正式组合模型")
    result_source = "正式组合模型"


# =========================
# 页面主体
# =========================
tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 经营总览", "🔎 数据洞察", "🧠 模型与验证", "📅 预测与计划", "⚠️ 误差诊断"]
)

with tab0:
    st.subheader("经营总览")
    st.markdown('<p class="section-note">一屏查看历史资金盘面、未来预测和需要优先关注的日期。</p>', unsafe_allow_html=True)

    history = read_csv_if_exists(DAILY_FEATURES_PATH)
    forecast = read_csv_if_exists(SUBMISSION_WITH_HEADER_PATH)
    validation = get_display_validation(result_source)

    if history is None or forecast is None:
        st.warning("总览所需数据尚未生成，请在左侧运行完整流程。")
    else:
        history["date"] = pd.to_datetime(history["date"])
        forecast["date"] = pd.to_datetime(forecast["report_date"].astype(str))
        forecast = add_net_flow(forecast)
        latest_30 = history.sort_values("date").tail(30)

        total_purchase = forecast["purchase"].sum()
        total_redeem = forecast["redeem"].sum()
        total_net = forecast["net_flow"].sum()
        peak_redeem_row = forecast.loc[forecast["redeem"].idxmax()]
        cols = st.columns(4)
        cols[0].metric("9月预计申购", format_amount(total_purchase))
        cols[1].metric("9月预计赎回", format_amount(total_redeem))
        cols[2].metric("9月预计净流量", format_amount(total_net), "净流入" if total_net >= 0 else "净流出")
        cols[3].metric("赎回峰值日", peak_redeem_row["date"].strftime("%m月%d日"), format_amount(peak_redeem_row["redeem"]))

        left, right = st.columns([1.65, 1])
        with left:
            overview_plot = forecast[["date", "purchase", "redeem"]].melt(
                "date", var_name="资金类型", value_name="金额"
            )
            fig = px.area(
                overview_plot,
                x="date",
                y="金额",
                color="资金类型",
                title="未来 30 天资金流动预测",
                color_discrete_map={"purchase": "#0f766e", "redeem": "#f97316"},
            )
            st.plotly_chart(base_figure(fig), use_container_width=True)
        with right:
            flow_fig = px.bar(
                forecast,
                x="date",
                y="net_flow",
                color="flow_status",
                title="每日净流入 / 净流出",
                color_discrete_map={"净流入": "#10b981", "净流出": "#ef4444"},
            )
            st.plotly_chart(base_figure(flow_fig), use_container_width=True)

        st.markdown("### 流动性关注清单")
        risk_threshold = float(latest_30["redeem"].quantile(0.75))
        risks = forecast[forecast["redeem"] >= risk_threshold].copy()
        risks["关注原因"] = "预计赎回高于近30日历史75分位"
        risks["建议准备金"] = (risks["redeem"] * 1.1).round().astype("int64")
        risks["日期"] = risks["date"].dt.strftime("%Y-%m-%d")
        if risks.empty:
            st.success("未来 30 天未发现高于近期阈值的赎回压力日。")
        else:
            st.dataframe(
                risks[["日期", "purchase", "redeem", "net_flow", "建议准备金", "关注原因"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "purchase": st.column_config.NumberColumn("预计申购", format="%,d"),
                    "redeem": st.column_config.NumberColumn("预计赎回", format="%,d"),
                    "net_flow": st.column_config.NumberColumn("预计净流量", format="%,d"),
                    "建议准备金": st.column_config.NumberColumn(format="%,d"),
                },
            )

        if validation is not None:
            metrics = evaluate_prediction(validation)
            st.caption(
                f"当前结果源：{result_source} · 内部模拟总分 {metrics['total_score']:.2f} · "
                f"申购 MAPE {metrics['purchase_mape']:.1%} · 赎回 MAPE {metrics['redeem_mape']:.1%}。"
                " 模拟分仅用于本地比较，并非官方成绩。"
            )

with tab1:
    st.subheader("数据洞察")
    st.markdown('<p class="section-note">按时间窗口探索资金趋势、周期规律与外部市场变量。</p>', unsafe_allow_html=True)
    df = read_csv_if_exists(DAILY_FEATURES_PATH)
    if df is None:
        st.warning("尚未生成 daily_features.csv，请先运行：python run_all.py")
    else:
        df["date"] = pd.to_datetime(df["date"])
        min_date, max_date = df["date"].min().date(), df["date"].max().date()
        selected_range = st.date_input(
            "分析日期范围",
            value=(max(min_date, (df["date"].max() - pd.Timedelta(days=120)).date()), max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if len(selected_range) == 2:
            start_date, end_date = selected_range
            view_df = df[df["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))].copy()
        else:
            view_df = df.copy()
        view_df = add_net_flow(view_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("区间天数", len(view_df))
        c2.metric("日均申购", format_amount(view_df["purchase"].mean()))
        c3.metric("日均赎回", format_amount(view_df["redeem"].mean()))
        c4.metric("累计净流量", format_amount(view_df["net_flow"].sum()))

        quality_report = read_csv_if_exists(DATA_QUALITY_REPORT_PATH)
        if quality_report is not None:
            st.markdown("### 数据质量校验")
            st.caption("余额一致性校验：tBalance = yBalance + total_purchase_amt - total_redeem_amt")
            st.dataframe(quality_report, use_container_width=True)

        fund_plot = view_df[["date", "purchase", "redeem"]].melt(id_vars="date", var_name="类型", value_name="金额")
        fund_fig = px.line(fund_plot, x="date", y="金额", color="类型", title="历史申购 / 赎回趋势")
        st.plotly_chart(base_figure(fund_fig), use_container_width=True)

        weekday_df = (
            view_df.assign(星期=view_df["date"].dt.dayofweek.map({0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}))
            .groupby("星期", as_index=False)[["purchase", "redeem", "net_flow"]]
            .mean()
        )
        weekday_order = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday_plot = weekday_df.melt("星期", value_vars=["purchase", "redeem"], var_name="类型", value_name="日均金额")
        weekday_fig = px.bar(
            weekday_plot, x="星期", y="日均金额", color="类型", barmode="group",
            category_orders={"星期": weekday_order}, title="星期周期画像"
        )
        st.plotly_chart(base_figure(weekday_fig, 350), use_container_width=True)

        yield_cols = [c for c in ["mfd_daily_yield", "mfd_7daily_yield"] if c in df.columns]
        if yield_cols:
            yield_plot = view_df[["date"] + yield_cols].melt("date", var_name="收益率字段", value_name="数值")
            st.plotly_chart(px.line(yield_plot, x="date", y="数值", color="收益率字段", title="收益率变化图"), use_container_width=True)

        shibor_cols = [c for c in df.columns if c.startswith("Interest_")]
        if shibor_cols:
            chosen_shibor = st.multiselect("选择要展示的 Shibor 指标", shibor_cols, default=shibor_cols[: min(3, len(shibor_cols))])
            if chosen_shibor:
                shibor_plot = view_df[["date"] + chosen_shibor].melt("date", var_name="Shibor字段", value_name="利率")
                st.plotly_chart(px.line(shibor_plot, x="date", y="利率", color="Shibor字段", title="Shibor 利率变化图"), use_container_width=True)

        with st.expander("查看与下载当前筛选数据"):
            st.dataframe(view_df, use_container_width=True, hide_index=True)
            st.download_button(
                "下载筛选数据 CSV",
                view_df.to_csv(index=False).encode("utf-8-sig"),
                "fund_flow_filtered.csv",
                mime="text/csv",
            )

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
    st.subheader("正式模型与验证")
    st.markdown('<p class="section-note">页面只展示当前正式组合模型，不再保留旧版模型切换入口。</p>', unsafe_allow_html=True)
    st.info(
        "模型先用多月份滚动回测选择稳定参数，再分别优化申购的月度总量/日内形状和赎回的节假日距离信号；"
        "最终采用线上验证表现最好的“月度形状申购 + 事件距离赎回”结构。"
    )

    flow_cols = st.columns(4)
    flow_cols[0].metric("选参区间", "2014年4—7月")
    flow_cols[1].metric("独立验证", "2014年8月")
    flow_cols[2].metric("申购模型", "月度形状")
    flow_cols[3].metric("赎回模型", "事件距离")

    display_valid = get_display_validation(result_source)
    if display_valid is None:
        st.warning("暂无正式组合模型验证结果，请先在左侧运行完整流程。")
    else:
        metrics = evaluate_prediction(display_valid)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("申购 MAPE", f"{metrics['purchase_mape']:.4f}")
        c2.metric("赎回 MAPE", f"{metrics['redeem_mape']:.4f}")
        c3.metric("模拟总分", f"{metrics['total_score']:.4f}")
        c4.metric("零分日", f"{int(display_valid['is_zero_score_day'].sum())} 天")

        purchase_plot = display_valid[["date", "purchase_true", "purchase_pred"]].melt("date", var_name="类型", value_name="金额")
        redeem_plot = display_valid[["date", "redeem_true", "redeem_pred"]].melt("date", var_name="类型", value_name="金额")
        left, right = st.columns(2)
        left.plotly_chart(px.line(purchase_plot, x="date", y="金额", color="类型", title="申购：真实值 vs 正式模型"), use_container_width=True)
        right.plotly_chart(px.line(redeem_plot, x="date", y="金额", color="类型", title="赎回：真实值 vs 正式模型"), use_container_width=True)

with tab3:
    st.subheader("预测与资金计划")
    st.markdown('<p class="section-note">把模型输出转换为可执行的每日资金安排与压力情景。</p>', unsafe_allow_html=True)
    pred = read_csv_if_exists(SUBMISSION_WITH_HEADER_PATH)
    if pred is None:
        st.warning("尚未生成预测结果文件，请先运行：python run_all.py")
    else:
        pred["date"] = pd.to_datetime(pred["report_date"].astype(str))
        pred = add_net_flow(pred)
        st.markdown("### 压力情景")
        col1, col2, col3 = st.columns([1, 1, 1.4])
        redeem_stress = col1.slider("赎回压力增幅", 0, 50, 10, 5, format="%d%%")
        reserve_buffer = col2.slider("准备金安全垫", 0, 30, 10, 5, format="%d%%")
        col3.info("情景参数只用于资金计划演算，不会修改模型结果或提交文件。")

        plan = pred.copy()
        plan["stress_redeem"] = plan["redeem"] * (1 + redeem_stress / 100)
        plan["stress_net_flow"] = plan["purchase"] - plan["stress_redeem"]
        plan["recommended_reserve"] = plan["stress_redeem"] * (1 + reserve_buffer / 100)
        plan["risk_level"] = pd.cut(
            plan["stress_net_flow"],
            bins=[-np.inf, -5e7, 0, np.inf],
            labels=["高", "中", "低"],
        ).astype(str)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("预计总申购", format_amount(plan["purchase"].sum()))
        k2.metric("压力情景总赎回", format_amount(plan["stress_redeem"].sum()), f"+{redeem_stress}%")
        k3.metric("最大单日准备金", format_amount(plan["recommended_reserve"].max()))
        k4.metric("高风险天数", f"{(plan['risk_level'] == '高').sum()} 天")

        pred_plot = plan[["date", "purchase", "redeem", "stress_redeem"]].melt("date", var_name="类型", value_name="金额")
        pred_fig = px.line(
            pred_plot,
            x="date",
            y="金额",
            color="类型",
            title="基准预测与压力情景",
            color_discrete_map={"purchase": "#0f766e", "redeem": "#f97316", "stress_redeem": "#dc2626"},
        )
        st.plotly_chart(base_figure(pred_fig), use_container_width=True)

        st.markdown("### 每日资金计划")
        plan_display = plan.copy()
        plan_display["date"] = plan_display["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            plan_display[["date", "purchase", "redeem", "net_flow", "stress_redeem", "recommended_reserve", "risk_level"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": "日期",
                "purchase": st.column_config.NumberColumn("预计申购", format="%d"),
                "redeem": st.column_config.NumberColumn("预计赎回", format="%d"),
                "net_flow": st.column_config.NumberColumn("基准净流量", format="%d"),
                "stress_redeem": st.column_config.NumberColumn("压力赎回", format="%.0f"),
                "recommended_reserve": st.column_config.NumberColumn("建议准备金", format="%.0f"),
                "risk_level": st.column_config.TextColumn("风险等级"),
            },
        )
        submit_bytes = pred[["report_date", "purchase", "redeem"]].to_csv(index=False, header=False).encode("utf-8-sig")
        plan_bytes = plan_display.to_csv(index=False).encode("utf-8-sig")
        d1, d2 = st.columns(2)
        d1.download_button("下载天池提交文件", submit_bytes, "tc_comp_predict_table.csv", mime="text/csv")
        d2.download_button("下载资金计划", plan_bytes, "fund_flow_plan.csv", mime="text/csv")

    display_valid = get_display_validation(result_source)
    if display_valid is not None:
        st.markdown("### 验证集真实值对比与每日误差/得分")
        show_cols = [
            "date", "purchase_true", "purchase_pred", "purchase_abs_error", "purchase_error", "purchase_score",
            "redeem_true", "redeem_pred", "redeem_abs_error", "redeem_error", "redeem_score", "daily_weighted_score",
        ]
        st.dataframe(display_valid[show_cols], use_container_width=True)

with tab4:
    st.subheader("误差诊断")
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
