import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    DAILY_FEATURES_PATH,
    ROOT_DIR,
    SUBMISSION_WITH_HEADER_PATH,
    VALIDATION_PRED_PATH,
)
from src.data_loader import load_all_tables, normalize_columns
from src.evaluate import build_daily_error_frame, evaluate_prediction
from src.train import train_and_evaluate

st.set_page_config(page_title="资金流入流出预测系统", layout="wide")
st.title("资金流入流出预测系统")
st.caption("数据探索 · 模型训练与评估 · 预测结果 · 误差分析")

SEX_MAP = {0: "女", 1: "男"}


@st.cache_data(show_spinner=False)
def read_csv_if_exists(path_str: str):
    path = Path(path_str)
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_user_profile():
    """返回 (dataframe, error_message)。成功时 error_message 为空字符串。"""
    try:
        user_profile, _, _, _ = load_all_tables()
        return normalize_columns(user_profile), ""
    except Exception as exc:
        return None, str(exc)


def history_to_frame(history, target_name: str) -> pd.DataFrame | None:
    if not history:
        return None
    # LightGBM: {'train': {'l2': [...]}, 'valid': {'l2': [...]}}
    frames = []
    for split_name, metrics in history.items():
        metric_name = "l2" if "l2" in metrics else next(iter(metrics))
        frames.append(pd.DataFrame({
            "iteration": list(range(1, len(metrics[metric_name]) + 1)),
            "loss": metrics[metric_name],
            "split": split_name,
            "target": target_name,
        }))
    return pd.concat(frames, ignore_index=True) if frames else None


def highlight_high_error(row, purchase_thr, redeem_thr):
    styles = [""] * len(row)
    cols = list(row.index)
    if "purchase_rel_error" in cols and row["purchase_rel_error"] >= purchase_thr:
        styles[cols.index("purchase_rel_error")] = "background-color: #ffc7ce; font-weight: 600"
        if "purchase_abs_error" in cols:
            styles[cols.index("purchase_abs_error")] = "background-color: #ffc7ce"
    if "redeem_rel_error" in cols and row["redeem_rel_error"] >= redeem_thr:
        styles[cols.index("redeem_rel_error")] = "background-color: #ffc7ce; font-weight: 600"
        if "redeem_abs_error" in cols:
            styles[cols.index("redeem_abs_error")] = "background-color: #ffc7ce"
    if "daily_score" in cols and row["daily_score"] <= 5:
        styles[cols.index("daily_score")] = "background-color: #ffeb9c; font-weight: 600"
    return styles


with st.sidebar:
    st.header("操作")
    st.write("项目根目录：", str(ROOT_DIR))
    if st.button("一键运行完整流程（预处理+训练+预测）"):
        with st.spinner("正在运行，请稍候..."):
            result = subprocess.run(
                [sys.executable, "run_all.py"],
                cwd=ROOT_DIR,
                text=True,
                capture_output=True,
            )
        st.code((result.stdout or "") + "\n" + (result.stderr or ""))
        if result.returncode == 0:
            st.success("运行完成，请刷新页面查看最新结果")
            st.cache_data.clear()
        else:
            st.error("运行失败，请查看上方日志")


tab1, tab2, tab3, tab4 = st.tabs(["数据探索", "模型训练与评估", "预测结果", "误差分析"])

# ---------------------------------------------------------------------------
# 1. 数据探索
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("历史申购 / 赎回趋势")
    df = read_csv_if_exists(str(DAILY_FEATURES_PATH))
    if df is None:
        st.warning("尚未生成 daily_features.csv，请先在侧边栏运行完整流程，或执行：python run_all.py")
    else:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("样本天数", f"{len(df)}")
        c2.metric("日均申购", f"{df['purchase'].mean():,.0f}")
        c3.metric("日均赎回", f"{df['redeem'].mean():,.0f}")
        c4.metric("活跃用户峰值", f"{int(df['user_count'].max()):,}" if "user_count" in df.columns else "-")

        plot_df = df[["date", "purchase", "redeem"]].melt(id_vars="date", var_name="类型", value_name="金额")
        st.plotly_chart(
            px.line(plot_df, x="date", y="金额", color="类型", title="每日申购/赎回总额趋势"),
            use_container_width=True,
        )

        if "user_count" in df.columns:
            st.plotly_chart(
                px.line(df, x="date", y="user_count", title="每日活跃用户数变化"),
                use_container_width=True,
            )

        st.subheader("收益率变化")
        yield_cols = [c for c in ["mfd_daily_yield", "mfd_7daily_yield"] if c in df.columns]
        if yield_cols:
            yield_plot = df[["date"] + yield_cols].melt("date", var_name="指标", value_name="数值")
            st.plotly_chart(
                px.line(yield_plot, x="date", y="数值", color="指标", title="余额宝收益率走势"),
                use_container_width=True,
            )
        else:
            st.info("特征表中暂无收益率字段。")

        st.subheader("银行间拆借利率（Shibor）")
        shibor_cols = [c for c in df.columns if c.startswith("Interest_")]
        if shibor_cols:
            selected = st.multiselect(
                "选择利率期限",
                shibor_cols,
                default=[c for c in ["Interest_O_N", "Interest_1_W", "Interest_1_M"] if c in shibor_cols],
            )
            if selected:
                shibor_plot = df[["date"] + selected].melt("date", var_name="期限", value_name="利率(%)")
                st.plotly_chart(
                    px.line(shibor_plot, x="date", y="利率(%)", color="期限", title="Shibor 利率变化"),
                    use_container_width=True,
                )
        else:
            st.info("特征表中暂无 Shibor 字段。")

        st.subheader("用户分布")
        profile, profile_error = load_user_profile()
        if profile_error:
            st.warning(f"用户画像加载失败：{profile_error}")
        elif profile is None:
            st.warning("未找到用户信息表。")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if "sex" in profile.columns:
                    sex_df = profile["sex"].map(SEX_MAP).fillna("未知").value_counts().reset_index()
                    sex_df.columns = ["性别", "用户数"]
                    st.plotly_chart(
                        px.pie(sex_df, names="性别", values="用户数", title="用户性别分布"),
                        use_container_width=True,
                    )
            with col_b:
                if "constellation" in profile.columns:
                    star_df = profile["constellation"].fillna("未知").value_counts().reset_index()
                    star_df.columns = ["星座", "用户数"]
                    st.plotly_chart(
                        px.bar(star_df, x="星座", y="用户数", title="用户星座分布"),
                        use_container_width=True,
                    )
            if "city" in profile.columns:
                city_df = profile["city"].value_counts().head(15).reset_index()
                city_df.columns = ["城市编码", "用户数"]
                st.plotly_chart(
                    px.bar(city_df, x="城市编码", y="用户数", title="用户城市 Top15（脱敏编码）"),
                    use_container_width=True,
                )

# ---------------------------------------------------------------------------
# 2. 模型训练与评估
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("选择模型与参数后训练")
    st.markdown(
        "评分说明：相对误差越小得分越高；**总分 = 申购得分 × 0.45 + 赎回得分 × 0.55**"
        "（赎回权重更高，建议优先优化赎回误差）。"
    )

    left, right = st.columns([1, 1])
    with left:
        model_type = st.selectbox("模型", ["LightGBM", "RandomForest"], index=0)
        n_estimators = st.slider("树的数量 n_estimators", 50, 800, 400, 50)
        max_depth = st.slider("最大深度 max_depth（-1 表示不限制，仅 LightGBM）", -1, 20, -1, 1)
    with right:
        learning_rate = st.slider("学习率 learning_rate（仅 LightGBM）", 0.01, 0.2, 0.03, 0.01)
        num_leaves = st.slider("叶子数 num_leaves（仅 LightGBM）", 8, 128, 31, 1)
        n_splits = st.slider("时序交叉验证折数", 2, 5, 3, 1)
        save_model = st.checkbox("训练后覆盖保存模型与验证结果", value=True)

    if st.button("开始训练并评估", type="primary"):
        params = {
            "model_type": "lightgbm" if model_type == "LightGBM" else "randomforest",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
        }
        with st.spinner("训练中，请稍候..."):
            try:
                train_result = train_and_evaluate(
                    model_params=params,
                    save=save_model,
                    run_cv=True,
                    n_splits=n_splits,
                )
                st.session_state["train_result"] = train_result
                st.cache_data.clear()
                st.success("训练完成")
            except Exception as exc:
                st.error(f"训练失败：{exc}")

    train_result = st.session_state.get("train_result")
    valid = read_csv_if_exists(str(VALIDATION_PRED_PATH))
    if train_result is not None:
        metrics = train_result["metrics"]
        source_label = "本次交互训练结果"
        valid_view = train_result["valid_result"].copy()
    elif valid is not None:
        valid_view = valid.copy()
        metrics = evaluate_prediction(valid_view)
        source_label = "已保存的验证集结果（可在上方重新训练）"
    else:
        valid_view = None
        metrics = None
        source_label = None

    if metrics is None:
        st.warning("尚无验证结果。请先训练，或运行 python run_all.py")
    else:
        st.caption(source_label)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("申购 MAPE", f"{metrics['purchase_mape']:.4f}")
        m2.metric("赎回 MAPE", f"{metrics['redeem_mape']:.4f}")
        m3.metric("申购得分", f"{metrics['purchase_score']:.3f}")
        m4.metric("赎回得分", f"{metrics['redeem_score']:.3f}")
        m5.metric("模拟总分", f"{metrics['total_score']:.3f}")

        valid_view["date"] = pd.to_datetime(valid_view["date"])
        purchase_plot = valid_view[["date", "purchase_true", "purchase_pred"]].melt(
            "date", var_name="类型", value_name="金额"
        )
        redeem_plot = valid_view[["date", "redeem_true", "redeem_pred"]].melt(
            "date", var_name="类型", value_name="金额"
        )
        st.plotly_chart(
            px.line(purchase_plot, x="date", y="金额", color="类型", title="申购：真实值 vs 预测值（8月验证集）"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(redeem_plot, x="date", y="金额", color="类型", title="赎回：真实值 vs 预测值（8月验证集）"),
            use_container_width=True,
        )

        st.subheader("训练损失曲线")
        if train_result is None:
            st.info("点击上方「开始训练并评估」后，可展示 LightGBM 的 train/valid 损失曲线。RandomForest 无线程损失记录。")
        else:
            loss_frames = []
            for hist, name in [
                (train_result.get("purchase_history"), "purchase"),
                (train_result.get("redeem_history"), "redeem"),
            ]:
                frame = history_to_frame(hist, name)
                if frame is not None:
                    loss_frames.append(frame)
            if not loss_frames:
                st.info("当前模型未提供迭代损失（例如 RandomForest）。可改用 LightGBM 查看损失曲线。")
            else:
                loss_df = pd.concat(loss_frames, ignore_index=True)
                st.plotly_chart(
                    px.line(
                        loss_df,
                        x="iteration",
                        y="loss",
                        color="split",
                        facet_col="target",
                        title="训练过程损失曲线（log1p 目标上的 L2）",
                    ),
                    use_container_width=True,
                )

        st.subheader("时序交叉验证结果")
        if train_result is not None and train_result.get("cv_results") is not None:
            cv_df = train_result["cv_results"]
            st.dataframe(cv_df, use_container_width=True)
            st.plotly_chart(
                px.bar(
                    cv_df.melt(id_vars="fold", value_vars=["purchase_mape", "redeem_mape", "total_score"],
                               var_name="指标", value_name="数值"),
                    x="fold",
                    y="数值",
                    color="指标",
                    barmode="group",
                    title="各折交叉验证指标",
                ),
                use_container_width=True,
            )
            st.caption(
                f"CV 平均总分：{cv_df['total_score'].mean():.3f}；"
                f"赎回 MAPE 均值：{cv_df['redeem_mape'].mean():.4f}"
            )
        else:
            st.info("重新训练后将展示时序交叉验证各折结果。")

# ---------------------------------------------------------------------------
# 3. 预测结果
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("2014年9月预测结果（提交文件）")
    pred = read_csv_if_exists(str(SUBMISSION_WITH_HEADER_PATH))
    if pred is None:
        st.warning("尚未生成预测结果文件，请先运行完整流程。")
    else:
        pred = pred.copy()
        pred["date"] = pd.to_datetime(pred["report_date"].astype(str))
        st.dataframe(pred[["report_date", "purchase", "redeem"]], use_container_width=True)
        pred_plot = pred[["date", "purchase", "redeem"]].melt("date", var_name="类型", value_name="金额")
        st.plotly_chart(
            px.line(pred_plot, x="date", y="金额", color="类型", title="2014年9月申购/赎回预测"),
            use_container_width=True,
        )
        submit_bytes = pred[["report_date", "purchase", "redeem"]].to_csv(index=False, header=False).encode("utf-8-sig")
        st.download_button("下载天池提交文件（无表头）", submit_bytes, "tc_comp_predict_table.csv")

    st.subheader("验证集对比与每日积分（有真实值）")
    st.caption("9 月无公开真实值；以下用 2014 年 8 月验证集展示预测 vs 真实、每日误差与总分。")
    valid = read_csv_if_exists(str(VALIDATION_PRED_PATH))
    if valid is None:
        st.warning("缺少 validation_prediction.csv")
    else:
        err_df = build_daily_error_frame(valid)
        metrics = evaluate_prediction(err_df)
        c1, c2, c3 = st.columns(3)
        c1.metric("验证集总分", f"{metrics['total_score']:.3f}")
        c2.metric("申购 MAPE", f"{metrics['purchase_mape']:.4f}")
        c3.metric("赎回 MAPE", f"{metrics['redeem_mape']:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=err_df["date"], y=err_df["purchase_true"], name="申购真实"))
        fig.add_trace(go.Scatter(x=err_df["date"], y=err_df["purchase_pred"], name="申购预测"))
        fig.add_trace(go.Scatter(x=err_df["date"], y=err_df["redeem_true"], name="赎回真实"))
        fig.add_trace(go.Scatter(x=err_df["date"], y=err_df["redeem_pred"], name="赎回预测"))
        fig.update_layout(title="8月验证集：真实值与预测值", xaxis_title="日期", yaxis_title="金额")
        st.plotly_chart(fig, use_container_width=True)

        score_plot = err_df[["date", "purchase_score", "redeem_score", "daily_score"]].melt(
            "date", var_name="得分项", value_name="得分"
        )
        st.plotly_chart(
            px.line(score_plot, x="date", y="得分", color="得分项", title="验证集每日模拟得分"),
            use_container_width=True,
        )
        show = err_df[[
            "date", "purchase_true", "purchase_pred", "purchase_abs_error", "purchase_rel_error",
            "redeem_true", "redeem_pred", "redeem_abs_error", "redeem_rel_error", "daily_score",
        ]].copy()
        show["purchase_rel_error"] = (show["purchase_rel_error"] * 100).round(2)
        show["redeem_rel_error"] = (show["redeem_rel_error"] * 100).round(2)
        show = show.rename(columns={
            "purchase_rel_error": "申购相对误差%",
            "redeem_rel_error": "赎回相对误差%",
            "purchase_abs_error": "申购绝对误差",
            "redeem_abs_error": "赎回绝对误差",
            "daily_score": "当日得分",
        })
        st.dataframe(show, use_container_width=True)

# ---------------------------------------------------------------------------
# 4. 误差分析
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("验证集每日误差与得分")
    valid = read_csv_if_exists(str(VALIDATION_PRED_PATH))
    if valid is None:
        st.warning("尚未生成 validation_prediction.csv，请先训练或运行完整流程。")
    else:
        err_df = build_daily_error_frame(valid)
        purchase_thr = float(err_df["purchase_rel_error"].quantile(0.8))
        redeem_thr = float(err_df["redeem_rel_error"].quantile(0.8))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("申购平均绝对误差", f"{err_df['purchase_abs_error'].mean():,.0f}")
        k2.metric("赎回平均绝对误差", f"{err_df['redeem_abs_error'].mean():,.0f}")
        k3.metric("申购平均相对误差", f"{err_df['purchase_rel_error'].mean():.2%}")
        k4.metric("赎回平均相对误差", f"{err_df['redeem_rel_error'].mean():.2%}")

        abs_plot = err_df.melt(
            id_vars="date",
            value_vars=["purchase_abs_error", "redeem_abs_error"],
            var_name="类型",
            value_name="绝对误差",
        )
        abs_plot["类型"] = abs_plot["类型"].map({
            "purchase_abs_error": "申购",
            "redeem_abs_error": "赎回",
        })
        st.plotly_chart(
            px.bar(abs_plot, x="date", y="绝对误差", color="类型", barmode="group", title="每日绝对误差"),
            use_container_width=True,
        )

        rel_plot = err_df.melt(
            id_vars="date",
            value_vars=["purchase_rel_error", "redeem_rel_error"],
            var_name="类型",
            value_name="相对误差",
        )
        rel_plot["类型"] = rel_plot["类型"].map({
            "purchase_rel_error": "申购",
            "redeem_rel_error": "赎回",
        })
        fig_rel = px.line(rel_plot, x="date", y="相对误差", color="类型", title="每日相对误差")
        fig_rel.add_hline(y=purchase_thr, line_dash="dot", annotation_text="申购高误差阈值(80%)")
        st.plotly_chart(fig_rel, use_container_width=True)

        st.plotly_chart(
            px.bar(err_df, x="date", y="daily_score", title="每日模拟得分（越低越需要关注）",
                   color="daily_score", color_continuous_scale="RdYlGn"),
            use_container_width=True,
        )

        table = err_df[[
            "date",
            "purchase_abs_error", "purchase_rel_error", "purchase_score",
            "redeem_abs_error", "redeem_rel_error", "redeem_score",
            "daily_score",
        ]].copy()
        st.dataframe(
            table.style.apply(
                lambda r: highlight_high_error(r, purchase_thr, redeem_thr),
                axis=1,
            ).format({
                "purchase_abs_error": "{:,.0f}",
                "redeem_abs_error": "{:,.0f}",
                "purchase_rel_error": "{:.2%}",
                "redeem_rel_error": "{:.2%}",
                "purchase_score": "{:.2f}",
                "redeem_score": "{:.2f}",
                "daily_score": "{:.2f}",
            }),
            use_container_width=True,
        )
        st.caption("红色高亮：相对误差位于验证集最高 20%；黄色高亮：当日得分 ≤ 5。")

        st.subheader("高误差日期与改进方向")
        worst = err_df.nsmallest(5, "daily_score")[
            ["date", "purchase_rel_error", "redeem_rel_error", "daily_score"]
        ].copy()
        worst["weekday"] = worst["date"].dt.day_name()
        st.dataframe(
            worst.assign(
                purchase_rel_error=lambda x: (x["purchase_rel_error"] * 100).round(2),
                redeem_rel_error=lambda x: (x["redeem_rel_error"] * 100).round(2),
            ).rename(columns={
                "purchase_rel_error": "申购相对误差%",
                "redeem_rel_error": "赎回相对误差%",
                "daily_score": "当日得分",
                "weekday": "星期",
            }),
            use_container_width=True,
        )

        redeem_worse = (err_df["redeem_rel_error"] > err_df["purchase_rel_error"]).mean()
        weekend_gap = (
            err_df.assign(is_weekend=lambda x: x["date"].dt.dayofweek >= 5)
            .groupby("is_weekend")["daily_score"]
            .mean()
        )
        tips = [
            f"约 {redeem_worse:.0%} 的日期赎回相对误差高于申购；赎回权重 55%，建议优先加强赎回滞后/滚动特征或单独加大赎回模型复杂度。",
            "高误差日多集中在周末或月初月末时，可增加节假日、发薪日、是否月初/月末交互特征。",
            "若损失曲线出现明显过拟合（train 持续下降而 valid 回升），可降低 num_leaves / n_estimators，或提高学习率稳定性。",
            "可在「模型训练与评估」页对比 LightGBM 与 RandomForest，并观察交叉验证中赎回 MAPE 是否同步下降。",
        ]
        if len(weekend_gap) == 2:
            tips.insert(
                1,
                f"周末平均得分 {weekend_gap.get(True, float('nan')):.2f}，工作日平均得分 {weekend_gap.get(False, float('nan')):.2f}。",
            )
        for tip in tips:
            st.markdown(f"- {tip}")
