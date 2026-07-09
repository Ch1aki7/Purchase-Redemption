# -*- coding: utf-8 -*-
"""
============================================================
 03_train_data_prepare.py
 训练数据准备脚本
 功能：
   1. 读取特征工程后的数据
   2. 划分训练集、验证集、测试集
   3. 特征标准化 / 归一化
   4. 处理类别不平衡
   5. 输出建模就绪的数据集
============================================================
"""

import os
import sys

# --- 自动检测并设置 JAVA_HOME ---
_JAVA_PATHS = [
    r"C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot",
    r"C:\Program Files\Java\jdk-17",
    r"C:\Program Files\Java\jdk-11",
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot",
]
if "JAVA_HOME" not in os.environ:
    for _p in _JAVA_PATHS:
        if os.path.isdir(_p):
            os.environ["JAVA_HOME"] = _p
            print(f"[INFO] 自动设置 JAVA_HOME = {_p}")
            break
    else:
        sys.exit("错误: 未找到Java，请安装JDK 17并设置JAVA_HOME环境变量")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, mean, stddev, min as spark_min, max as spark_max,
    when, to_date, datediff, dayofweek, lit, array, monotonically_increasing_id
)
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, MinMaxScaler, StringIndexer,
    OneHotEncoder, Imputer, PCA
)
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
import time


def init_spark(app_name="TrainDataPreparation"):
    """初始化Spark会话"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_raw_and_engineer_from_scratch(spark, data_dir="Purchase Redemption Data"):
    """
    从原始数据直接构建训练数据集
    这个方法整合了数据读取、聚合、合并、特征工程，
    避免依赖02脚本的输出文件
    """
    from pyspark.sql.functions import (
        countDistinct, sum as spark_sum, avg as spark_avg, stddev,
        lag, expr, coalesce, to_date, dayofweek, dayofmonth, month,
        year, weekofyear, quarter, sin, cos
    )
    from pyspark.sql.window import Window

    print("=" * 60)
    print("Step 1: 从原始数据构建每日聚合表")
    print("=" * 60)

    # 读取数据
    user_balance = spark.read \
        .option("header", "true").option("inferSchema", "true") \
        .csv(f"{data_dir}/user_balance_table.csv")

    day_share_interest = spark.read \
        .option("header", "true").option("inferSchema", "true") \
        .csv(f"{data_dir}/mfd_day_share_interest.csv")

    bank_shibor = spark.read \
        .option("header", "true").option("inferSchema", "true") \
        .csv(f"{data_dir}/mfd_bank_shibor.csv")

    # ============ 每日聚合 ============
    print("  聚合用户数据到每日级别...")
    daily_df = user_balance.groupBy("report_date").agg(
        countDistinct("user_id").alias("active_users"),
        countDistinct(when(col("total_purchase_amt") > 0,
                           col("user_id"))).alias("purchase_users"),
        countDistinct(when(col("total_redeem_amt") > 0,
                           col("user_id"))).alias("redeem_users"),

        spark_sum("total_purchase_amt").alias("total_purchase"),
        spark_sum("total_redeem_amt").alias("total_redeem"),
        spark_sum("direct_purchase_amt").alias("direct_purchase"),
        spark_sum("purchase_bal_amt").alias("purchase_bal"),
        spark_sum("purchase_bank_amt").alias("purchase_bank"),
        spark_sum("share_amt").alias("total_share"),
        spark_sum("consume_amt").alias("total_consume"),
        spark_sum("transfer_amt").alias("total_transfer"),
        spark_sum("tftobal_amt").alias("total_tftobal"),
        spark_sum("tftocard_amt").alias("total_tftocard"),
        spark_sum("category1").alias("cat1"),
        spark_sum("category2").alias("cat2"),
        spark_sum("category3").alias("cat3"),
        spark_sum("category4").alias("cat4"),
        spark_avg("tBalance").alias("avg_balance"),
    )

    # ============ 合并收益率和Shibor ============
    print("  合并收益率和Shibor利率数据...")
    daily_df = daily_df.join(
        day_share_interest,
        daily_df["report_date"] == day_share_interest["mfd_date"],
        "left"
    ).drop("mfd_date")

    daily_df = daily_df.join(
        bank_shibor,
        daily_df["report_date"] == bank_shibor["mfd_date"],
        "left"
    ).drop("mfd_date")

    # 前向填充缺失值
    raw_window = Window.orderBy("report_date").rowsBetween(
        Window.unboundedPreceding, 0
    )
    fill_cols = ["mfd_daily_yield", "mfd_7daily_yield",
                 "Interest_O_N", "Interest_1_W", "Interest_2_W",
                 "Interest_1_M", "Interest_3_M", "Interest_6_M",
                 "Interest_9_M", "Interest_1_Y"]
    for c in fill_cols:
        daily_df = daily_df.withColumn(
            c, expr(f"last({c}, true) over (order by report_date "
                    f"rows between unbounded preceding and current row)")
        )

    # 确保日期排序
    daily_df = daily_df.orderBy("report_date")

    return daily_df


def create_all_features(daily_df):
    """
    在每日聚合数据基础上构造全部特征
    """
    from pyspark.sql.functions import (
        lag, expr, coalesce, to_date, dayofweek, dayofmonth, month,
        year, weekofyear, quarter, when, avg as spark_avg,
        stddev, max as spark_max, min as spark_min
    )
    from pyspark.sql.window import Window

    print("\n" + "=" * 60)
    print("Step 2: 构造全部特征")
    print("=" * 60)

    window_spec = Window.orderBy("report_date")

    # ---------- 时间特征 ----------
    print("  [时间特征]")
    daily_df = daily_df \
        .withColumn("date", to_date(col("report_date").cast("string"), "yyyyMMdd")) \
        .withColumn("year", year("date")) \
        .withColumn("month", month("date")) \
        .withColumn("day", dayofmonth("date")) \
        .withColumn("day_of_week", dayofweek("date")) \
        .withColumn("week_of_year", weekofyear("date")) \
        .withColumn("quarter", quarter("date")) \
        .withColumn("is_weekend",
                    when(dayofweek("date").isin([1, 7]), 1).otherwise(0)) \
        .withColumn("is_month_start",
                    when(dayofmonth("date") <= 3, 1).otherwise(0)) \
        .withColumn("is_month_end",
                    when(dayofmonth("date") >= 28, 1).otherwise(0)) \
        .withColumn("dow_sin", expr("sin(2 * 3.1415926535 * day_of_week / 7.0)")) \
        .withColumn("dow_cos", expr("cos(2 * 3.1415926535 * day_of_week / 7.0)")) \
        .withColumn("month_sin", expr("sin(2 * 3.1415926535 * month / 12.0)")) \
        .withColumn("month_cos", expr("cos(2 * 3.1415926535 * month / 12.0)"))

    # ---------- 比率特征 ----------
    print("  [比率特征]")
    daily_df = daily_df \
        .withColumn("purchase_user_ratio",
                    col("purchase_users") / col("active_users")) \
        .withColumn("redeem_user_ratio",
                    col("redeem_users") / col("active_users")) \
        .withColumn("avg_purchase_per_user",
                    col("total_purchase") / col("active_users")) \
        .withColumn("avg_redeem_per_user",
                    col("total_redeem") / col("active_users")) \
        .withColumn("net_flow",
                    col("total_purchase") - col("total_redeem")) \
        .withColumn("consume_ratio",
                    when(col("total_redeem") > 0,
                         col("total_consume") / col("total_redeem")).otherwise(0)) \
        .withColumn("transfer_ratio",
                    when(col("total_redeem") > 0,
                         col("total_transfer") / col("total_redeem")).otherwise(0)) \
        .withColumn("direct_purchase_ratio",
                    when(col("total_purchase") > 0,
                         col("direct_purchase") / col("total_purchase")).otherwise(0))

    # ---------- 滞后特征 ----------
    print("  [滞后特征]")
    lag_targets = {
        "total_purchase": [1, 2, 3, 7, 14, 21, 28],
        "total_redeem": [1, 2, 3, 7, 14, 21, 28],
        "active_users": [1, 7, 28],
        "total_share": [1, 7, 28],
        "total_consume": [1, 7, 28],
        "net_flow": [1, 7, 28],
    }
    for col_name, lags in lag_targets.items():
        for d in lags:
            daily_df = daily_df.withColumn(
                f"{col_name}_lag_{d}",
                lag(col(col_name), d).over(window_spec)
            )

    # 环比变化
    daily_df = daily_df \
        .withColumn("purchase_change_1d",
                    (col("total_purchase") - col("total_purchase_lag_1"))
                    / coalesce(col("total_purchase_lag_1"), lit(1))) \
        .withColumn("redeem_change_1d",
                    (col("total_redeem") - col("total_redeem_lag_1"))
                    / coalesce(col("total_redeem_lag_1"), lit(1)))

    # ---------- 滚动窗口特征 ----------
    print("  [滚动窗口特征]")
    roll_targets = ["total_purchase", "total_redeem", "active_users",
                    "total_share", "total_consume", "net_flow"]
    for c in roll_targets:
        for w in [7, 14, 30]:
            win = Window.orderBy("report_date").rowsBetween(-w, -1)
            daily_df = daily_df \
                .withColumn(f"{c}_roll_{w}d_avg",
                            spark_avg(col(c)).over(win)) \
                .withColumn(f"{c}_roll_{w}d_std",
                            stddev(col(c)).over(win)) \
                .withColumn(f"{c}_roll_{w}d_max",
                            spark_max(col(c)).over(win)) \
                .withColumn(f"{c}_roll_{w}d_min",
                            spark_min(col(c)).over(win))

    # ---------- 趋势 / 动量特征 ----------
    print("  [趋势特征]")
    for c in ["total_purchase", "total_redeem", "total_consume"]:
        daily_df = daily_df.withColumn(
            f"{c}_momentum_7v30",
            when(col(f"{c}_roll_30d_avg") > 0,
                 col(f"{c}_roll_7d_avg") / col(f"{c}_roll_30d_avg")).otherwise(1.0)
        )

    # ---------- 收益率特征 ----------
    print("  [收益率特征]")
    daily_df = daily_df \
        .withColumn("yield_change",
                    col("mfd_daily_yield") - lag("mfd_daily_yield", 1).over(window_spec)) \
        .withColumn("yield_7d_change",
                    col("mfd_7daily_yield") - lag("mfd_7daily_yield", 1).over(window_spec))
    for d in [1, 3, 7]:
        daily_df = daily_df.withColumn(
            f"yield_lag_{d}",
            lag("mfd_daily_yield", d).over(window_spec)
        )

    # ---------- Shibor特征 ----------
    print("  [Shibor特征]")
    daily_df = daily_df \
        .withColumn("shibor_spread_1M_O_N",
                    col("Interest_1_M") - col("Interest_O_N")) \
        .withColumn("shibor_spread_1Y_O_N",
                    col("Interest_1_Y") - col("Interest_O_N")) \
        .withColumn("shibor_slope",
                    col("Interest_1_Y") - col("Interest_O_N")) \
        .withColumn("shibor_on_change",
                    col("Interest_O_N") - lag("Interest_O_N", 1).over(window_spec))

    # ---------- 填充缺失值 ----------
    print("  [填充缺失值]")
    numeric_cols = [f.name for f in daily_df.schema.fields
                    if f.dataType.typeName() in ('double', 'long', 'integer', 'float')]
    for c in numeric_cols:
        daily_df = daily_df.withColumn(c, coalesce(col(c), lit(0)))

    print(f"  总特征数（含目标列）: {len(daily_df.columns)}")

    return daily_df


def split_dataset(daily_df):
    """
    划分训练集、验证集、测试集
    - 训练集: 2013-07-01 ~ 2014-07-31
    - 验证集: 2014-08-01 ~ 2014-08-31
    - 预测目标: 2014-09-01 ~ 2014-09-30（测试集期间无实际数据）
    """
    print("\n" + "=" * 60)
    print("Step 3: 划分训练/验证集")
    print("=" * 60)

    train_df = daily_df.filter(col("report_date") <= 20140731)
    val_df = daily_df.filter(
        (col("report_date") >= 20140801) & (col("report_date") <= 20140831)
    )

    train_count = train_df.count()
    val_count = val_df.count()

    print(f"  训练集（~2014-07-31）: {train_count:,} 行")
    print(f"  验证集（2014-08）: {val_count:,} 行")
    print(f"  训练集日期范围: ", end="")
    train_df.agg(
        spark_min("report_date").alias("min"),
        spark_max("report_date").alias("max")
    ).show()
    print(f"  验证集日期范围: ", end="")
    val_df.agg(
        spark_min("report_date").alias("min"),
        spark_max("report_date").alias("max")
    ).show()

    return train_df, val_df


def compute_statistics(df, label="数据集"):
    """
    输出数据集的描述性统计
    """
    print(f"\n--- {label} 关键指标统计 ---")

    stats = df.select(
        "total_purchase", "total_redeem", "active_users",
        "total_share", "total_consume", "net_flow"
    ).describe()

    stats.show()

    # 目标变量的分布概要
    target_stats = df.select(
        spark_min("total_purchase").alias("min_purchase"),
        spark_max("total_purchase").alias("max_purchase"),
        mean("total_purchase").alias("mean_purchase"),
        stddev("total_purchase").alias("std_purchase"),
        spark_min("total_redeem").alias("min_redeem"),
        spark_max("total_redeem").alias("max_redeem"),
        mean("total_redeem").alias("mean_redeem"),
        stddev("total_redeem").alias("std_redeem"),
    ).collect()[0]

    print(f"\n  目标变量统计:")
    print(f"  total_purchase: min={target_stats['min_purchase']:,.0f}, "
          f"max={target_stats['max_purchase']:,.0f}, "
          f"mean={target_stats['mean_purchase']:,.0f}, "
          f"std={target_stats['std_purchase']:,.0f}")
    print(f"  total_redeem:   min={target_stats['min_redeem']:,.0f}, "
          f"max={target_stats['max_redeem']:,.0f}, "
          f"mean={target_stats['mean_redeem']:,.0f}, "
          f"std={target_stats['std_redeem']:,.0f}")


def prepare_modeling_dataset(train_df, val_df):
    """
    准备可用于建模的数据集：
    1. 选择特征列
    2. 使用VectorAssembler组装特征向量
    3. 标准化（可选）
    """
    print("\n" + "=" * 60)
    print("Step 4: 准备建模数据集")
    print("=" * 60)

    # 定义特征列（排除非特征列）
    exclude_cols = {
        "report_date", "date",
        "total_purchase", "total_redeem",  # 目标变量
        "year", "month_label",  # 非数值/冗余列
    }

    # 选择数值特征列
    feature_cols = [c for c in train_df.columns
                    if c not in exclude_cols
                    and c in train_df.columns]

    # 确保所有列都是数值类型
    valid_feature_cols = []
    for c in feature_cols:
        dtype = train_df.schema[c].dataType.typeName()
        if dtype in ('double', 'long', 'integer', 'float', 'decimal'):
            valid_feature_cols.append(c)

    print(f"  建模特征数: {len(valid_feature_cols)}")
    print(f"  前10个特征: {valid_feature_cols[:10]}")

    # ====== 方法1: 使用VectorAssembler（用于Spark MLlib） ======
    print("\n  [VectorAssembler 组装特征向量]")
    assembler = VectorAssembler(
        inputCols=valid_feature_cols,
        outputCol="features_raw",
        handleInvalid="skip"
    )

    train_assembled = assembler.transform(train_df) \
        .select("report_date", "total_purchase", "total_redeem", "features_raw")
    val_assembled = assembler.transform(val_df) \
        .select("report_date", "total_purchase", "total_redeem", "features_raw")

    # ====== 方法2: 标准化特征向量 ======
    print("\n  [StandardScaler 标准化]")
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features_scaled",
        withStd=True,
        withMean=True
    )

    scaler_model = scaler.fit(train_assembled)
    train_scaled = scaler_model.transform(train_assembled)
    val_scaled = scaler_model.transform(val_assembled)

    print(f"  训练集标准化后样例:")
    train_scaled.select("report_date", "total_purchase",
                        "total_redeem").show(5)

    return train_scaled, val_scaled, valid_feature_cols, scaler_model


def analyze_feature_correlation(train_df, feature_cols, top_n=30):
    """
    分析特征与目标变量的相关性
    """
    print("\n" + "=" * 60)
    print("Step 5: 特征相关性分析")
    print("=" * 60)

    # 由于Spark的Correlation计算需要向量，我们只分析关键特征
    key_cols = [c for c in feature_cols
                if not c.endswith(('_lag_21', '_lag_28', '_roll_30d_std',
                                   '_roll_30d_max', '_roll_30d_min'))
                and c in train_df.columns]

    # 限制列数避免计算溢出
    if len(key_cols) > 50:
        # 优先选择非滞后非滚动的基础特征 + 最重要的滞后/滚动特征
        priority_cols = [c for c in key_cols
                         if 'lag' not in c and 'roll' not in c and 'momentum' not in c]
        lag_short = [c for c in key_cols
                     if '_lag_1' in c or '_lag_2' in c or '_lag_3' in c or '_lag_7' in c]
        roll_short = [c for c in key_cols if '_roll_7d_avg' in c or '_roll_14d_avg' in c]
        key_cols = priority_cols + lag_short + roll_short

    print(f"  计算 {len(key_cols)} 个特征的相关性矩阵...")

    # 组装特征向量用于相关性计算
    tmp_assembler = VectorAssembler(
        inputCols=key_cols,
        outputCol="corr_features",
        handleInvalid="skip"
    )
    corr_df = tmp_assembler.transform(train_df).select("corr_features")

    # 计算皮尔逊相关系数矩阵
    try:
        corr_matrix = Correlation.corr(corr_df, "corr_features", "pearson").head()
        print(f"  相关性矩阵大小: {corr_matrix[0].numRows} x {corr_matrix[0].numCols}")

        # 提取与目标变量（total_purchase在feature_cols中的位置）相关性最高的特征
        # 这里由于feature_cols包含了全部特征，我们用简化的方式输出
        corr_array = corr_matrix[0].toArray()

        # 找到target相关的最高相关性（target在这key_cols列表中的位置）
        target_idx_map = {}
        for t in ["total_purchase", "total_redeem"]:
            for i, c in enumerate(key_cols):
                if t in c.lower() and 'lag' in c.split('_')[-2:]:
                    pass  # 跳过，因为target不在key_cols中
        # 简化：输出矩阵中有显著相关性的特征对
        print("\n  高相关性特征对（|r| > 0.9，可能共线）:")
        high_corr_pairs = []
        n = min(corr_array.shape[0], 100)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_array[i][j]) > 0.9:
                    if i < len(key_cols) and j < len(key_cols):
                        high_corr_pairs.append(
                            (key_cols[i], key_cols[j], corr_array[i][j])
                        )

        if high_corr_pairs:
            # 只显示前20对
            for c1, c2, r in high_corr_pairs[:20]:
                print(f"    {c1} <-> {c2}: {r:.4f}")
        else:
            print("    未发现高度共线的特征对")

    except Exception as e:
        print(f"  相关性计算出错（可忽略，特征过多时常见）: {e}")
        print("  建议在建模时使用特征重要性方法筛选特征")


def analyze_target_trend(train_df, val_df):
    """
    分析目标变量在训练集和验证集上的趋势差异
    """
    print("\n" + "=" * 60)
    print("Step 6: 目标变量趋势分析")
    print("=" * 60)

    # 按月统计
    for name, df in [("Train", train_df), ("Val", val_df)]:
        print(f"\n  [{name}集按月统计]")
        monthly = df.groupBy("month").agg(
            mean("total_purchase").alias("avg_purchase"),
            mean("total_redeem").alias("avg_redeem"),
            spark_min("total_purchase").alias("min_purchase"),
            spark_max("total_purchase").alias("max_purchase"),
        ).orderBy("month")
        monthly.show()

    # 按星期几统计
    print(f"\n  [训练集按星期几统计]")
    dow_stats = train_df.groupBy("day_of_week").agg(
        mean("total_purchase").alias("avg_purchase"),
        mean("total_redeem").alias("avg_redeem"),
        mean("active_users").alias("avg_active_users"),
    ).orderBy("day_of_week")
    dow_stats.show()


def prepare_prediction_template(spark):
    """
    准备预测用的模板 —— 需要预测2014年9月的每一天
    注意：9月的特征需要用之前的数据构造（lag和rolling特征）
    实际预测时，需要用8月31日及之前的历史数据逐日滚动构造特征
    """
    print("\n" + "=" * 60)
    print("Step 7: 准备预测模板")
    print("=" * 60)

    # 生成2014年9月所有日期（共30天）
    sep_dates = [20140901 + i for i in range(30)]

    pred_template = spark.createDataFrame(
        [(d,) for d in sep_dates], ["report_date"]
    )

    print(f"  预测日期范围: {sep_dates[0]} ~ {sep_dates[-1]}")
    print(f"  预测天数: {len(sep_dates)} 天")

    pred_template.show(5)

    return pred_template, sep_dates


def save_datasets(train_df, val_df, train_scaled, val_scaled,
                  feature_cols, output_path="modeling_output"):
    """
    保存处理后的数据集（使用Pandas写CSV，避免Hadoop依赖）
    """
    import os as _os
    _os.makedirs(output_path, exist_ok=True)

    print("\n" + "=" * 60)
    print("Step 8: 保存数据集")
    print("=" * 60)

    # 原始特征数据（用于tree-based模型如GBDT）
    print("  保存原始特征数据...")
    train_pdf = train_df.toPandas()
    train_pdf.to_csv(f"{output_path}/train_data.csv", index=False, encoding="utf-8-sig")
    print(f"    训练集: {len(train_pdf)} 行 x {len(train_pdf.columns)} 列")

    val_pdf = val_df.toPandas()
    val_pdf.to_csv(f"{output_path}/val_data.csv", index=False, encoding="utf-8-sig")
    print(f"    验证集: {len(val_pdf)} 行 x {len(val_pdf.columns)} 列")

    # 标准化后的数据（用于线性模型如Ridge，或神经网络）
    print("  保存标准化数据...")
    train_s_pdf = train_scaled.toPandas()
    train_s_pdf.to_csv(f"{output_path}/train_scaled.csv", index=False, encoding="utf-8-sig")
    print(f"    训练集(标准化): {len(train_s_pdf)} 行")

    val_s_pdf = val_scaled.toPandas()
    val_s_pdf.to_csv(f"{output_path}/val_scaled.csv", index=False, encoding="utf-8-sig")
    print(f"    验证集(标准化): {len(val_s_pdf)} 行")

    print(f"  数据已保存至: {output_path}/")


def print_summary(train_count, val_count, feature_count):
    """打印数据准备摘要"""
    print("\n" + "=" * 60)
    print("数据准备完成！摘要如下：")
    print("=" * 60)
    print(f"  训练集样本数: {train_count:,}")
    print(f"  验证集样本数: {val_count:,}")
    print(f"  特征数量:     {feature_count}")
    print(f"  目标变量:     total_purchase, total_redeem")
    print(f"  预测任务:     预测2014年9月1日-30日的每日总申购额和总赎回额")
    print(f"  评估指标:     相对误差 → 得分映射 → 申购*45% + 赎回*55%")
    print(f"\n  注: 2014年9月的预测特征需要从8月31日的历史数据滚动构造")
    print("=" * 60)


def main():
    spark = init_spark()
    start_time = time.time()

    data_dir = "Purchase Redemption Data"

    # Step 1: 构建每日聚合表
    daily_df = read_raw_and_engineer_from_scratch(spark, data_dir)

    # Step 2: 构造特征
    daily_df = create_all_features(daily_df)

    # 缓存数据
    daily_df.cache()

    # Step 3: 划分数据集
    train_df, val_df = split_dataset(daily_df)

    # Step 4: 描述性统计
    compute_statistics(train_df, "训练集")
    compute_statistics(val_df, "验证集")

    # Step 5: 目标变量趋势
    analyze_target_trend(train_df, val_df)

    # Step 6: 准备建模数据集
    train_scaled, val_scaled, feature_cols, scaler_model = \
        prepare_modeling_dataset(train_df, val_df)

    # Step 7: 相关性分析
    analyze_feature_correlation(train_df, feature_cols)

    # Step 8: 预测模板
    pred_template, sep_dates = prepare_prediction_template(spark)

    # Step 9: 保存
    save_datasets(train_df, val_df, train_scaled, val_scaled, feature_cols)

    # Summary
    print_summary(train_df.count(), val_df.count(), len(feature_cols))

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.2f} 秒")

    spark.stop()


if __name__ == "__main__":
    main()
