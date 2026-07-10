import subprocess
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import DAILY_FEATURES_PATH, VALIDATION_PRED_PATH, SUBMISSION_PATH, SUBMISSION_WITH_HEADER_PATH, ROOT_DIR
from src.evaluate import evaluate_prediction, relative_error

st.set_page_config(page_title="资金流入流出预测系统", layout="wide")
st.title("资金流入流出预测系统")
st.caption("最小可跑通课程项目：数据探索、模型评估、9月预测结果展示、误差分析")

with st.sidebar:
    st.header("操作")
    st.write("当前项目根目录：", str(ROOT_DIR))
    if st.button("一键运行完整流程"):
        with st.spinner("正在运行，请稍候..."):
            result = subprocess.run([sys.executable, "run_all.py"], cwd=ROOT_DIR, text=True, capture_output=True)
        st.code(result.stdout + "\n" + result.stderr)
        if result.returncode == 0:
            st.success("运行完成")
        else:
            st.error("运行失败，请查看上方日志")


def read_csv_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


tab1, tab2, tab3, tab4 = st.tabs(["数据探索", "模型评估", "预测结果", "误差分析"])

with tab1:
    st.subheader("历史每日申购/赎回趋势")
    df = read_csv_if_exists(DAILY_FEATURES_PATH)
    if df is None:
        st.warning("尚未生成 daily_features.csv，请先运行：python run_all.py")
    else:
        df["date"] = pd.to_datetime(df["date"])
        st.dataframe(df.head(20), use_container_width=True)
        plot_df = df[["date", "purchase", "redeem"]].melt(id_vars="date", var_name="类型", value_name="金额")
        fig = px.line(plot_df, x="date", y="金额", color="类型", title="每日申购/赎回总额趋势")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("8月验证集预测对比")
    valid = read_csv_if_exists(VALIDATION_PRED_PATH)
    if valid is None:
        st.warning("尚未生成 validation_prediction.csv，请先运行：python run_all.py")
    else:
        valid["date"] = pd.to_datetime(valid["date"])
        metrics = evaluate_prediction(valid)
        c1, c2, c3 = st.columns(3)
        c1.metric("申购 MAPE", f"{metrics['purchase_mape']:.4f}")
        c2.metric("赎回 MAPE", f"{metrics['redeem_mape']:.4f}")
        c3.metric("模拟总分", f"{metrics['total_score']:.4f}")

        purchase_plot = valid[["date", "purchase_true", "purchase_pred"]].melt("date", var_name="类型", value_name="金额")
        redeem_plot = valid[["date", "redeem_true", "redeem_pred"]].melt("date", var_name="类型", value_name="金额")
        st.plotly_chart(px.line(purchase_plot, x="date", y="金额", color="类型", title="申购：真实值 vs 预测值"), use_container_width=True)
        st.plotly_chart(px.line(redeem_plot, x="date", y="金额", color="类型", title="赎回：真实值 vs 预测值"), use_container_width=True)

with tab3:
    st.subheader("2014年9月预测结果")
    pred = read_csv_if_exists(SUBMISSION_WITH_HEADER_PATH)
    if pred is None:
        st.warning("尚未生成预测结果文件，请先运行：python run_all.py")
    else:
        pred["date"] = pd.to_datetime(pred["report_date"].astype(str))
        st.dataframe(pred[["report_date", "purchase", "redeem"]], use_container_width=True)
        pred_plot = pred[["date", "purchase", "redeem"]].melt("date", var_name="类型", value_name="金额")
        st.plotly_chart(px.line(pred_plot, x="date", y="金额", color="类型", title="2014年9月申购/赎回预测"), use_container_width=True)
        submit_bytes = pred[["report_date", "purchase", "redeem"]].to_csv(index=False, header=False).encode("utf-8-sig")
        st.download_button("下载天池提交文件（无表头）", submit_bytes, "tc_comp_predict_table.csv")

with tab4:
    st.subheader("验证集每日误差分析")
    valid = read_csv_if_exists(VALIDATION_PRED_PATH)
    if valid is None:
        st.warning("尚未生成 validation_prediction.csv，请先运行：python run_all.py")
    else:
        valid["purchase_error"] = relative_error(valid["purchase_true"], valid["purchase_pred"])
        valid["redeem_error"] = relative_error(valid["redeem_true"], valid["redeem_pred"])
        show_cols = ["date", "purchase_true", "purchase_pred", "purchase_error", "redeem_true", "redeem_pred", "redeem_error"]
        st.dataframe(
            valid[show_cols].style.background_gradient(subset=["purchase_error", "redeem_error"], cmap="Reds"),
            use_container_width=True,
        )
