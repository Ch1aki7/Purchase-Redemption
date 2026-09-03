import subprocess
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import (
    DAILY_FEATURES_PATH,
    VALIDATION_PRED_PATH,
    SUBMISSION_PATH,
    SUBMISSION_WITH_HEADER_PATH,
    ROOT_DIR,
)
from src.evaluate import evaluate_prediction, relative_error

st.set_page_config(page_title="资金流入流出预测系统", layout="wide")
st.title("资金流入流出预测系统")
st.caption("课程项目展示：数据探索、模型配置、验证评估、9月预测、误差与异常点分析")

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

    st.divider()
    st.markdown("**当前模型摘要**")
    st.write("多种子 LightGBM 集成")
    st.write("验证方式：8月滚动预测")
    st.write("评分权重：申购45%，赎回55%")


def read_csv_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def add_error_columns(valid: pd.DataFrame) -> pd.DataFrame:
    valid = valid.copy()
    valid["date"] = pd.to_datetime(valid["date"])
    valid["purchase_error"] = relative_error(valid["purchase_true"], valid["purchase_pred"])
    valid["redeem_error"] = relative_error(valid["redeem_true"], valid["redeem_pred"])
    valid["purchase_direction"] = valid.apply(
        lambda r: "高估" if r["purchase_pred"] > r["purchase_true"] else "低估", axis=1
    )
    valid["redeem_direction"] = valid.apply(
        lambda r: "高估" if r["redeem_pred"] > r["redeem_true"] else "低估", axis=1
    )
    valid["is_zero_score_day"] = (valid["purchase_error"] > 0.3) | (valid["redeem_error"] > 0.3)
    valid["异常类型"] = valid["date"].apply(classify_anomaly)
    return valid


def classify_anomaly(date):
    date = pd.Timestamp(date)
    day = date.day
    dow = date.dayofweek
    if day == 1:
        return "月初异常"
    if day >= date.days_in_month - 2:
        return "月末异常"
    if dow == 6:
        return "周日波动"
    if date.strftime("%m-%d") == "08-08":
        return "孤立高值"
    if day >= date.days_in_month - 4:
        return "月末附近"
    return "普通日期"


OPTIMIZATION_HISTORY = pd.DataFrame([
    {"阶段": "初始基线", "说明": "基础时间特征 + 单一LightGBM", "总分": 4.96, "申购MAPE": "16.68%", "赎回MAPE": "18.41%", "零分日": "10天"},
    {"阶段": "第一/二轮", "说明": "扩展特征 + 早期集成调参（后确认存在泄露/验证易化）", "总分": 8.14, "申购MAPE": "6.01%", "赎回MAPE": "5.22%", "零分日": "0天（虚高）"},
    {"阶段": "第三轮", "说明": "删除泄露特征 + 验证集改为滚动预测", "总分": 5.15, "申购MAPE": "15.55%", "赎回MAPE": "16.45%", "零分日": "9天"},
    {"阶段": "第四轮", "说明": "多种子LightGBM + 小学习率 + smooth=0.99", "总分": 5.19, "申购MAPE": "14.86%", "赎回MAPE": "16.72%", "零分日": "9天"},
    {"阶段": "第五轮", "说明": "月末3天后处理：申购×1.2、赎回×1.3", "总分": 5.43, "申购MAPE": "13.55%", "赎回MAPE": "15.93%", "零分日": "6天"},
])

COURSE_REQUIREMENTS = pd.DataFrame([
    {"课程要求": "整合四组数据表", "项目实现": "读取用户申购赎回、用户画像、收益率、Shibor四张表", "状态": "已完成"},
    {"课程要求": "数据清洗与特征工程", "项目实现": "日期解析、日级聚合、收益率/利率合并、113个历史特征", "状态": "已完成"},
    {"课程要求": "余额一致性约束", "项目实现": "文档说明 tBalance = yBalance + purchase - redeem，保留后续显式校验入口", "状态": "已说明"},
    {"课程要求": "预测未来30天", "项目实现": "滚动预测2014-09-01至2014-09-30申购和赎回", "状态": "已完成"},
    {"课程要求": "重点优化赎回55%权重", "项目实现": "赎回独立模型，月末赎回×1.3，零分日专项分析", "状态": "已完成"},
    {"课程要求": "数据探索模块", "项目实现": "历史申购/赎回趋势图与原始特征表展示", "状态": "已完成"},
    {"课程要求": "模型训练与评估模块", "项目实现": "展示模型配置、优化路径、验证集指标", "状态": "已完成"},
    {"课程要求": "预测结果展示模块", "项目实现": "9月预测表格、折线图、无表头提交文件下载", "状态": "已完成"},
    {"课程要求": "误差分析模块", "项目实现": "每日误差、高估/低估方向、零分日和异常类型", "状态": "已完成"},
])

MODEL_CONFIG = pd.DataFrame([
    {"项目": "模型", "当前配置": "3个不同随机种子的 LightGBM 平均"},
    {"项目": "目标变换", "当前配置": "训练 log1p，预测 expm1 还原"},
    {"项目": "learning_rate", "当前配置": "0.001"},
    {"项目": "n_estimators", "当前配置": "10000，配合100轮早停"},
    {"项目": "特征数量", "当前配置": "约113个日级特征，全部基于历史值或外生变量"},
    {"项目": "验证方式", "当前配置": "2014年8月逐日滚动预测，预测值回填为下一天lag"},
    {"项目": "平滑修正", "当前配置": "模型预测×0.99 + 近7日均值×0.01"},
    {"项目": "月末后处理", "当前配置": "月末3天申购×1.2、赎回×1.3"},
    {"项目": "RandomForest", "当前配置": "保留为降级接口，最终权重为0，未参与当前预测"},
])


tab1, tab2, tab3, tab4, tab5 = st.tabs(["数据探索", "模型评估", "预测结果", "误差分析", "课程要求对照"])

with tab1:
    st.subheader("历史每日申购/赎回趋势")
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
        plot_df = df[["date", "purchase", "redeem"]].melt(id_vars="date", var_name="类型", value_name="金额")
        fig = px.line(plot_df, x="date", y="金额", color="类型", title="每日申购/赎回总额趋势")
        st.plotly_chart(fig, use_container_width=True)
        st.info("用户画像表已读取作为数据资产；由于本任务预测日级总量，静态用户画像难以稳定映射到未来每日资金流，当前主模型未直接使用用户画像特征。")

with tab2:
    st.subheader("模型训练与验证评估")
    st.markdown("### 当前模型配置")
    st.dataframe(MODEL_CONFIG, use_container_width=True)

    st.markdown("### 多轮优化路径")
    st.dataframe(OPTIMIZATION_HISTORY, use_container_width=True)
    fig = px.line(OPTIMIZATION_HISTORY, x="阶段", y="总分", markers=True, title="验证集模拟总分优化路径")
    st.plotly_chart(fig, use_container_width=True)

    valid = read_csv_if_exists(VALIDATION_PRED_PATH)
    if valid is None:
        st.warning("尚未生成 validation_prediction.csv，请先运行：python run_all.py")
    else:
        valid = add_error_columns(valid)
        metrics = evaluate_prediction(valid)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("申购 MAPE", f"{metrics['purchase_mape']:.4f}")
        c2.metric("赎回 MAPE", f"{metrics['redeem_mape']:.4f}")
        c3.metric("模拟总分", f"{metrics['total_score']:.4f}")
        c4.metric("零分日", f"{int(valid['is_zero_score_day'].sum())} 天")

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
        valid = add_error_columns(valid)
        show_cols = [
            "date",
            "异常类型",
            "purchase_true",
            "purchase_pred",
            "purchase_error",
            "purchase_direction",
            "redeem_true",
            "redeem_pred",
            "redeem_error",
            "redeem_direction",
        ]
        st.markdown("### 每日误差表")
        st.dataframe(
            valid[show_cols].style.background_gradient(subset=["purchase_error", "redeem_error"], cmap="Reds"),
            use_container_width=True,
        )

        st.markdown("### 零分日（误差 > 30%）")
        zero_days = valid[valid["is_zero_score_day"]].copy()
        if zero_days.empty:
            st.success("当前验证集没有误差超过30%的零分日。")
        else:
            st.dataframe(zero_days[show_cols], use_container_width=True)
            zero_plot = zero_days[["date", "purchase_error", "redeem_error"]].melt("date", var_name="误差类型", value_name="相对误差")
            st.plotly_chart(px.bar(zero_plot, x="date", y="相对误差", color="误差类型", barmode="group", title="零分日相对误差"), use_container_width=True)

        st.markdown("### 异常点解释")
        st.info(
            "剩余零分日主要来自三类异常：月初1日申购异常高、月末附近赎回异常高、周日资金波动方向不一致。"
            "这些日期偏离历史均值较大，且方向不统一，因此未继续使用更激进固定系数，以避免针对验证集过拟合。"
        )

with tab5:
    st.subheader("课程指导书要求对照")
    st.dataframe(COURSE_REQUIREMENTS, use_container_width=True)
    st.markdown("### 评分说明")
    st.write("指导书只公开 error=0 得10分、error>0.3 得0分、得分函数单调递减；具体公式不公布。")
    st.write("本项目使用线性近似公式 `score = max(0, 10 × (1 - error / 0.3))` 做本地模拟评估，文档中均标注为模拟分数，非官方成绩。")
    st.markdown("### 合规性说明")
    st.write("验证集采用与9月预测一致的滚动预测逻辑，不使用8月真实历史构造未来lag；所有目标相关特征均来自历史值或外生变量，避免数据泄露。")
