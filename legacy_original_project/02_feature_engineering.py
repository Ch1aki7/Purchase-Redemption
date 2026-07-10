# -*- coding: utf-8 -*-
"""
============================================================
 02_feature_engineering.py
 特征工程脚本
 功能：
   1. 将用户级数据聚合为每日总量数据
   2. 合并收益率表、Shibor利率表
   3. 构造时间特征、滞后特征、滚动窗口特征
   4. 输出特征工程后的数据集
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
    col, count, countDistinct, sum as spark_sum, avg, stddev,
    max as spark_max, min as spark_min,
    to_date, dayofweek, dayofmonth, month, year, weekofyear, quarter,
    when, lit, lag, lead, row_number, round as spark_round,
    coalesce, expr, monotonically_increasing_id
)
from pyspark.sql.types import DoubleType, LongType, IntegerType
from pyspark.sql.window import Window
import time


def init_spark(app_name="FeatureEngineering"):
    """初始化Spark会话"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_raw_data(spark, data_dir="Purchase Redemption Data"):
    """读取原始数据"""
    print("读取原始数据...")

    user_balance = spark.read \
        .option("header", "true").option("inferSchema", "true") \
        .csv(f"{data_dir}/user_balance_table.csv")

    day_share_interest = spark.read \
        .option("header", "true").option("inferSchema", "true") \
        .csv(f"{data_dir}/mfd_day_share_interest.csv")

    bank_shibor = spark.read \
        .option("header", "true").option("inferSchema", "true") \
        .csv(f"{data_dir}/mfd_bank_shibor.csv")

    print(f"  user_balance: {user_balance.count():,} 行")
    print(f"  day_share_interest: {day_share_interest.count()} 行")
    print(f"  bank_shibor: {bank_shibor.count()} 行")

    return user_balance, day_share_interest, bank_shibor


def aggregate_daily(user_balance):
    """
    将用户级数据聚合为每日总量数据
    对每个日期，汇总所有用户的申购、赎回、消费、转出等指标
    """
    print("\n" + "=" * 60)
    print("Step 1: 用户级 -> 每日级聚合")
    print("=" * 60)

    daily_agg = user_balance.groupBy("report_date").agg(
        # 用户行为统计
        countDistinct("user_id").alias("active_users"),
        countDistinct(
            when(col("total_purchase_amt") > 0, col("user_id"))
        ).alias("purchase_users"),
        countDistinct(
            when(col("total_redeem_amt") > 0, col("user_id"))
        ).alias("redeem_users"),
        countDistinct(
            when(col("consume_amt") > 0, col("user_id"))
        ).alias("consume_users"),
        countDistinct(
            when(col("transfer_amt") > 0, col("user_id"))
        ).alias("transfer_users"),

        # 申购相关（目标变量）
        spark_sum("total_purchase_amt").alias("total_purchase"),
        spark_sum("direct_purchase_amt").alias("direct_purchase"),
        spark_sum("purchase_bal_amt").alias("purchase_bal"),
        spark_sum("purchase_bank_amt").alias("purchase_bank"),

        # 赎回相关（目标变量）
        spark_sum("total_redeem_amt").alias("total_redeem"),
        spark_sum("consume_amt").alias("total_consume"),
        spark_sum("transfer_amt").alias("total_transfer"),
        spark_sum("tftobal_amt").alias("total_tftobal"),
        spark_sum("tftocard_amt").alias("total_tftocard"),

        # 收益
        spark_sum("share_amt").alias("total_share"),

        # 消费类目
        spark_sum("category1").alias("cat1_total"),
        spark_sum("category2").alias("cat2_total"),
        spark_sum("category3").alias("cat3_total"),
        spark_sum("category4").alias("cat4_total"),

        # 余额统计
        spark_sum("tBalance").alias("total_balance"),
        avg("tBalance").alias("avg_balance"),
        stddev("tBalance").alias("std_balance"),
        spark_max("tBalance").alias("max_balance"),
        spark_sum("yBalance").alias("total_yesterday_balance"),

        # 人均指标
        (spark_sum("total_purchase_amt") / countDistinct("user_id")).alias("avg_purchase_per_user"),
        (spark_sum("total_redeem_amt") / countDistinct("user_id")).alias("avg_redeem_per_user"),
        (spark_sum("consume_amt") / countDistinct("user_id")).alias("avg_consume_per_user"),
    )

    # 派生比率特征
    daily_agg = daily_agg \
        .withColumn("purchase_user_ratio",
                    col("purchase_users") / col("active_users")) \
        .withColumn("redeem_user_ratio",
                    col("redeem_users") / col("active_users")) \
        .withColumn("consume_ratio_in_redeem",
                    when(col("total_redeem") > 0,
                         col("total_consume") / col("total_redeem")).otherwise(0)) \
        .withColumn("transfer_ratio_in_redeem",
                    when(col("total_redeem") > 0,
                         col("total_transfer") / col("total_redeem")).otherwise(0)) \
        .withColumn("direct_purchase_ratio",
                    when(col("total_purchase") > 0,
                         col("direct_purchase") / col("total_purchase")).otherwise(0)) \
        .withColumn("purchase_bank_ratio",
                    when(col("total_purchase") > 0,
                         col("purchase_bank") / col("total_purchase")).otherwise(0)) \
        .withColumn("net_flow",
                    col("total_purchase") - col("total_redeem"))

    # 确保按日期排序
    daily_agg = daily_agg.orderBy("report_date")

    print(f"  聚合后行数: {daily_agg.count()}")
    print("  样例数据:")
    daily_agg.select("report_date", "active_users", "total_purchase",
                     "total_redeem", "total_share").show(5)

    return daily_agg


def merge_external_data(daily_agg, day_share_interest, bank_shibor):
    """
    合并收益率表和Shibor利率表
    """
    print("\n" + "=" * 60)
    print("Step 2: 合并外部数据（收益率 + Shibor）")
    print("=" * 60)

    # 统一日期列名并join
    daily_with_yield = daily_agg.join(
        day_share_interest,
        daily_agg["report_date"] == day_share_interest["mfd_date"],
        "left"
    ).drop("mfd_date")

    daily_all = daily_with_yield.join(
        bank_shibor,
        daily_with_yield["report_date"] == bank_shibor["mfd_date"],
        "left"
    ).drop("mfd_date")

    # 检查合并后的缺失情况
    null_yield = daily_all.filter(col("mfd_daily_yield").isNull()).count()
    null_shibor = daily_all.filter(col("Interest_O_N").isNull()).count()
    print(f"  收益率缺失行数: {null_yield}")
    print(f"  Shibor缺失行数: {null_shibor}")

    # 前向填充缺失值（用最近的有效值填充）
    if null_yield > 0 or null_shibor > 0:
        print("  使用前向填充处理缺失值...")
        yield_cols = ["mfd_daily_yield", "mfd_7daily_yield"]
        shibor_cols = ["Interest_O_N", "Interest_1_W", "Interest_2_W",
                       "Interest_1_M", "Interest_3_M", "Interest_6_M",
                       "Interest_9_M", "Interest_1_Y"]

        all_fill_cols = yield_cols + shibor_cols
        window_spec = Window.orderBy("report_date") \
            .rowsBetween(Window.unboundedPreceding, 0)

        for c in all_fill_cols:
            # 使用last(ignoreNulls=True)进行前向填充
            daily_all = daily_all.withColumn(
                c,
                expr(f"""
                    last({c}, true)
                    over (order by report_date
                          rows between unbounded preceding and current row)
                """)
            )

    print("  合并后样例:")
    daily_all.select(
        "report_date", "total_purchase", "total_redeem",
        "mfd_daily_yield", "Interest_O_N"
    ).show(5)

    return daily_all


def create_time_features(daily_df):
    """
    构造时间特征：
    - 基础日期特征：年、月、日、星期几、是否周末、是否月初/月末
    - 周期性编码：星期几的sin/cos编码
    """
    print("\n" + "=" * 60)
    print("Step 3: 构造时间特征")
    print("=" * 60)

    # 将日期转为Date类型
    daily_df = daily_df.withColumn(
        "date", to_date(col("report_date").cast("string"), "yyyyMMdd")
    )

    # 基础时间特征
    daily_df = daily_df \
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
        .withColumn("is_monday",
                    when(dayofweek("date") == 2, 1).otherwise(0)) \
        .withColumn("is_friday",
                    when(dayofweek("date") == 6, 1).otherwise(0))

    # 周期性编码：day_of_week (1=Sunday, 7=Saturday -> 用sin/cos编码)
    daily_df = daily_df \
        .withColumn("dow_sin",
                    expr("sin(2 * 3.1415926535 * day_of_week / 7.0)")) \
        .withColumn("dow_cos",
                    expr("cos(2 * 3.1415926535 * day_of_week / 7.0)")) \
        .withColumn("month_sin",
                    expr("sin(2 * 3.1415926535 * month / 12.0)")) \
        .withColumn("month_cos",
                    expr("cos(2 * 3.1415926535 * month / 12.0)")) \
        .withColumn("day_sin",
                    expr("sin(2 * 3.1415926535 * day / 31.0)")) \
        .withColumn("day_cos",
                    expr("cos(2 * 3.1415926535 * day / 31.0)"))

    # 节假日效应标记：国庆节前（9月底可能有赎回潮）、春节等
    daily_df = daily_df \
        .withColumn("month_label",
                    when(col("month") == 1, "Jan")
                    .when(col("month") == 2, "Feb")
                    .when(col("month") == 3, "Mar")
                    .when(col("month") == 4, "Apr")
                    .when(col("month") == 5, "May")
                    .when(col("month") == 6, "Jun")
                    .when(col("month") == 7, "Jul")
                    .when(col("month") == 8, "Aug")
                    .when(col("month") == 9, "Sep")
                    .when(col("month") == 10, "Oct")
                    .when(col("month") == 11, "Nov")
                    .when(col("month") == 12, "Dec")
                    .otherwise("Unknown"))

    print("  时间特征示例:")
    daily_df.select("report_date", "day_of_week", "is_weekend",
                    "month", "quarter", "dow_sin", "dow_cos").show(5)

    return daily_df


def create_lag_features(daily_df, target_cols, lag_days=[1, 2, 3, 7, 14, 21, 28, 30]):
    """
    为关键指标构造滞后特征（过去N天的值）
    使用Spark Window函数实现
    """
    print("\n" + "=" * 60)
    print("Step 4: 构造滞后特征")
    print("=" * 60)

    window_spec = Window.orderBy("report_date")

    for c in target_cols:
        for lag_d in lag_days:
            daily_df = daily_df.withColumn(
                f"{c}_lag_{lag_d}",
                lag(col(c), lag_d).over(window_spec)
            )

    # 申购赎回的日环比
    daily_df = daily_df \
        .withColumn("purchase_change_1d",
                    (col("total_purchase") - col("total_purchase_lag_1"))
                    / coalesce(col("total_purchase_lag_1"), lit(1))) \
        .withColumn("redeem_change_1d",
                    (col("total_redeem") - col("total_redeem_lag_1"))
                    / coalesce(col("total_redeem_lag_1"), lit(1)))

    print(f"  为 {len(target_cols)} 个指标各构造了 {len(lag_days)} 个滞后特征")
    lag_cols = [f"{c}_lag_{d}" for c in target_cols for d in lag_days]
    print(f"  共生成 {len(lag_cols)} 个滞后特征")

    return daily_df


def create_rolling_features(daily_df, target_cols, windows=[7, 14, 30]):
    """
    构造滚动窗口统计特征：
    - 过去N天的均值、标准差、最小值、最大值
    """
    print("\n" + "=" * 60)
    print("Step 5: 构造滚动窗口特征")
    print("=" * 60)

    for c in target_cols:
        for w in windows:
            window_spec = Window.orderBy("report_date") \
                .rowsBetween(-w, -1)

            daily_df = daily_df \
                .withColumn(f"{c}_roll_{w}d_avg",
                            avg(col(c)).over(window_spec)) \
                .withColumn(f"{c}_roll_{w}d_std",
                            stddev(col(c)).over(window_spec)) \
                .withColumn(f"{c}_roll_{w}d_max",
                            spark_max(col(c)).over(window_spec)) \
                .withColumn(f"{c}_roll_{w}d_min",
                            spark_min(col(c)).over(window_spec))

    print(f"  为 {len(target_cols)} 个指标各构造了 {len(windows)} 个窗口特征")
    print(f"  每个窗口包含: avg, std, max, min")

    return daily_df


def create_trend_features(daily_df):
    """
    构造趋势特征：
    - 近7日 vs 近30日均值比值（短期 vs 长期趋势）
    - 近7日均值 vs 前7日均值变化
    """
    print("\n" + "=" * 60)
    print("Step 6: 构造趋势特征")
    print("=" * 60)

    for c in ["total_purchase", "total_redeem", "active_users",
              "total_share", "total_consume"]:
        # 7日均值 / 30日均值 = 短期/长期动量
        daily_df = daily_df.withColumn(
            f"{c}_momentum_7v30",
            when(col(f"{c}_roll_7d_avg") > 0,
                 col(f"{c}_roll_7d_avg") / col(f"{c}_roll_30d_avg")
                 ).otherwise(1.0)
        )

        # 7日趋势：最近7日均值 vs 再前7日均值
        window_prev = Window.orderBy("report_date").rowsBetween(-14, -8)
        daily_df = daily_df \
            .withColumn(f"{c}_avg_prev7", avg(col(c)).over(window_prev)) \
            .withColumn(f"{c}_trend_7v7",
                        when(col(f"{c}_avg_prev7") > 0,
                             col(f"{c}_roll_7d_avg") / col(f"{c}_avg_prev7")
                             ).otherwise(1.0)) \
            .drop(f"{c}_avg_prev7")

    print("  趋势特征示例（申购）:")
    daily_df.select(
        "report_date", "total_purchase",
        "total_purchase_momentum_7v30", "total_purchase_trend_7v7"
    ).show(5)

    return daily_df


def create_yield_shibor_features(daily_df):
    """
    构造收益率和Shibor相关特征：
    - 收益率的变化
    - 不同期限Shibor的利差
    - 收益率和Shibor的滞后特征
    """
    print("\n" + "=" * 60)
    print("Step 7: 构造收益率和利率特征")
    print("=" * 60)

    window_spec = Window.orderBy("report_date")

    # 收益率变化（差分）
    daily_df = daily_df \
        .withColumn("yield_change_1d",
                    col("mfd_daily_yield") - lag("mfd_daily_yield", 1).over(window_spec)) \
        .withColumn("yield_7d_change",
                    col("mfd_7daily_yield") - lag("mfd_7daily_yield", 1).over(window_spec))

    # 收益率滞后特征
    for lag_d in [1, 3, 7]:
        daily_df = daily_df \
            .withColumn(f"yield_lag_{lag_d}",
                        lag("mfd_daily_yield", lag_d).over(window_spec)) \
            .withColumn(f"yield_7d_lag_{lag_d}",
                        lag("mfd_7daily_yield", lag_d).over(window_spec))

    # Shibor利差特征（反映市场流动性预期）
    daily_df = daily_df \
        .withColumn("shibor_spread_1M_O_N",
                    col("Interest_1_M") - col("Interest_O_N")) \
        .withColumn("shibor_spread_1Y_O_N",
                    col("Interest_1_Y") - col("Interest_O_N")) \
        .withColumn("shibor_spread_3M_1M",
                    col("Interest_3_M") - col("Interest_1_M")) \
        .withColumn("shibor_slope",
                    col("Interest_1_Y") - col("Interest_O_N"))

    # Shibor变化
    daily_df = daily_df \
        .withColumn("shibor_on_change",
                    col("Interest_O_N") - lag("Interest_O_N", 1).over(window_spec)) \
        .withColumn("shibor_1w_change",
                    col("Interest_1_W") - lag("Interest_1_W", 1).over(window_spec))

    print("  收益率/Shibor特征样例:")
    daily_df.select(
        "report_date", "mfd_daily_yield", "yield_change_1d",
        "shibor_spread_1M_O_N", "shibor_slope"
    ).show(5)

    return daily_df


def fill_missing_lag_values(daily_df):
    """
    填充因滞后/窗口计算产生的缺失值
    由于前期数据不足，lag和rolling特征在开头会有NULL
    """
    print("\n" + "=" * 60)
    print("Step 8: 处理滞后和窗口的缺失值")
    print("=" * 60)

    # 统计每列的缺失数
    total = daily_df.count()
    null_stats = []
    for c in daily_df.columns:
        null_count = daily_df.filter(col(c).isNull()).count()
        if null_count > 0:
            null_stats.append((c, null_count))

    print(f"  有缺失值的列数: {len(null_stats)}")
    null_stats.sort(key=lambda x: -x[1])
    for c, cnt in null_stats[:10]:
        print(f"    {c}: {cnt} 行缺失 ({cnt/total*100:.1f}%)")

    # 前向填充（对于有先后顺序的时间序列数据）
    # 对于lag特征的开头缺失，可以用0或均值填充
    numeric_cols = [f.name for f in daily_df.schema.fields
                    if f.dataType.typeName() in ('double', 'long', 'integer', 'float')]

    for c in numeric_cols:
        # 对每个数值列，用0填充NULL（lag特征在开头没有历史数据时合理为0）
        daily_df = daily_df.withColumn(c, coalesce(col(c), lit(0)))

    print(f"  已用0填充所有数值型缺失值")

    return daily_df


def add_user_profile_features(spark, daily_df, data_dir):
    """
    从用户画像表构造每日的用户画像聚合特征
    注：需要user_balance和user_profile join
    """
    print("\n" + "=" * 60)
    print("Step 9: 用户画像特征（按日聚合）")
    print("=" * 60)

    # 读取用户画像
    user_profile = spark.read \
        .option("header", "true").option("inferSchema", "true") \
        .csv(f"{data_dir}/user_profile_table.csv")

    # 读取用户交易数据
    user_balance = spark.read \
        .option("header", "true").option("inferSchema", "true") \
        .csv(f"{data_dir}/user_balance_table.csv")

    # Join获取用户画像
    user_with_profile = user_balance.join(user_profile, on="user_id", how="left")

    # 按日期和性别聚合
    gender_daily = user_with_profile.groupBy("report_date", "sex").agg(
        spark_sum("total_purchase_amt").alias("gender_purchase"),
        spark_sum("total_redeem_amt").alias("gender_redeem"),
        countDistinct("user_id").alias("gender_users")
    )

    # 透视：男性
    male_stats = gender_daily.filter(col("sex") == 1).select(
        "report_date",
        col("gender_purchase").alias("male_purchase"),
        col("gender_redeem").alias("male_redeem"),
        col("gender_users").alias("male_users")
    )

    # 透视：女性
    female_stats = gender_daily.filter(col("sex") == 0).select(
        "report_date",
        col("gender_purchase").alias("female_purchase"),
        col("gender_redeem").alias("female_redeem"),
        col("gender_users").alias("female_users")
    )

    # 合并回每日数据
    daily_df = daily_df \
        .join(male_stats, on="report_date", how="left") \
        .join(female_stats, on="report_date", how="left")

    # 性别比率
    daily_df = daily_df \
        .withColumn("male_purchase_ratio",
                    when(col("total_purchase") > 0,
                         col("male_purchase") / col("total_purchase")).otherwise(0)) \
        .withColumn("male_user_ratio",
                    when(col("active_users") > 0,
                         col("male_users") / col("active_users")).otherwise(0))

    # 填充NULL
    for c in ["male_purchase", "male_redeem", "male_users",
              "female_purchase", "female_redeem", "female_users"]:
        daily_df = daily_df.withColumn(c, coalesce(col(c), lit(0)))

    print("  用户画像特征已添加")
    daily_df.select("report_date", "male_purchase_ratio", "male_user_ratio").show(5)

    return daily_df


def select_final_features(daily_df):
    """
    整理最终特征列，去除中间计算列
    """
    print("\n" + "=" * 60)
    print("Step 10: 整理最终特征集")
    print("=" * 60)

    # 定义需要保留的特征列
    base_cols = ["report_date"]
    target_cols = ["total_purchase", "total_redeem"]

    # 用户行为特征
    behavior_cols = [
        "active_users", "purchase_users", "redeem_users",
        "consume_users", "transfer_users",
        "purchase_user_ratio", "redeem_user_ratio",
        "avg_purchase_per_user", "avg_redeem_per_user", "avg_consume_per_user",
    ]

    # 金额细分特征
    amount_cols = [
        "direct_purchase", "purchase_bal", "purchase_bank",
        "total_consume", "total_transfer", "total_tftobal", "total_tftocard",
        "total_share", "cat1_total", "cat2_total", "cat3_total", "cat4_total",
        "consume_ratio_in_redeem", "transfer_ratio_in_redeem",
        "direct_purchase_ratio", "purchase_bank_ratio", "net_flow"
    ]

    # 余额特征
    balance_cols = ["total_balance", "avg_balance", "std_balance",
                    "max_balance", "total_yesterday_balance"]

    # 时间特征
    time_cols = [
        "year", "month", "day", "day_of_week", "week_of_year", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
        "is_monday", "is_friday",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "day_sin", "day_cos"
    ]

    # 收益率特征
    yield_cols = [
        "mfd_daily_yield", "mfd_7daily_yield",
        "yield_change_1d", "yield_7d_change",
        "yield_lag_1", "yield_lag_3", "yield_lag_7",
        "yield_7d_lag_1", "yield_7d_lag_3", "yield_7d_lag_7",
    ]

    # Shibor特征
    shibor_cols = [
        "Interest_O_N", "Interest_1_W", "Interest_2_W",
        "Interest_1_M", "Interest_3_M", "Interest_6_M",
        "Interest_9_M", "Interest_1_Y",
        "shibor_spread_1M_O_N", "shibor_spread_1Y_O_N",
        "shibor_spread_3M_1M", "shibor_slope",
        "shibor_on_change", "shibor_1w_change",
    ]

    # 用户画像特征
    profile_cols = [
        "male_purchase", "male_redeem", "male_users",
        "female_purchase", "female_redeem", "female_users",
        "male_purchase_ratio", "male_user_ratio",
    ]

    # 滞后和滚动特征（动态生成列名）
    lag_targets = ["total_purchase", "total_redeem", "active_users",
                   "total_share", "total_consume", "direct_purchase"]
    lag_days = [1, 2, 3, 7, 14, 21, 28, 30]
    lag_cols = [f"{c}_lag_{d}" for c in lag_targets for d in lag_days]

    roll_cols = []
    roll_windows = [7, 14, 30]
    for c in ["total_purchase", "total_redeem", "active_users",
              "total_share", "total_consume"]:
        for w in roll_windows:
            for stat in ["avg", "std", "max", "min"]:
                roll_cols.append(f"{c}_roll_{w}d_{stat}")

    trend_cols = []
    for c in ["total_purchase", "total_redeem", "active_users",
              "total_share", "total_consume"]:
        trend_cols.append(f"{c}_momentum_7v30")
        trend_cols.append(f"{c}_trend_7v7")

    change_cols = ["purchase_change_1d", "redeem_change_1d"]

    # 汇总所有需要的特征列
    all_feature_cols = (
        base_cols + target_cols +
        behavior_cols + amount_cols + balance_cols +
        time_cols + yield_cols + shibor_cols + profile_cols +
        lag_cols + roll_cols + trend_cols + change_cols
    )

    # 只保留存在的列
    existing_cols = [c for c in all_feature_cols if c in daily_df.columns]
    final_df = daily_df.select(existing_cols)

    print(f"  最终特征数: {len(existing_cols) - 3}")  # 减去report_date和2个target
    print(f"  总列数: {len(existing_cols)}")
    print(f"  总行数: {final_df.count()}")

    return final_df


def save_engineered_data(final_df, output_path="feature_engineering_output"):
    """保存特征工程结果（使用Pandas写CSV，避免Hadoop依赖）"""
    print("\n" + "=" * 60)
    print("Step 11: 保存特征工程结果")
    print("=" * 60)

    import os as _os
    _os.makedirs(output_path, exist_ok=True)

    csv_path = f"{output_path}/engineered_features.csv"
    pdf = final_df.toPandas()
    pdf.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  特征工程数据已保存至: {csv_path} ({len(pdf)} 行 x {len(pdf.columns)} 列)")

    # 简单统计摘要
    print(f"\n  最终数据集概况:")
    print(f"  行数: {final_df.count()}")
    print(f"  列数: {len(final_df.columns)}")
    print(f"  目标列: total_purchase, total_redeem")


def main():
    spark = init_spark()
    start_time = time.time()

    data_dir = "Purchase Redemption Data"

    # 1. 读取原始数据
    user_balance, day_share_interest, bank_shibor = read_raw_data(spark, data_dir)

    # 2. 聚合到每日级别
    daily_agg = aggregate_daily(user_balance)

    # 3. 合并外部数据
    daily_df = merge_external_data(daily_agg, day_share_interest, bank_shibor)

    # 4. 时间特征
    daily_df = create_time_features(daily_df)

    # 5. 滞后特征
    lag_targets = ["total_purchase", "total_redeem", "active_users",
                   "total_share", "total_consume", "direct_purchase"]
    daily_df = create_lag_features(daily_df, lag_targets)

    # 6. 滚动窗口特征
    roll_targets = ["total_purchase", "total_redeem", "active_users",
                    "total_share", "total_consume"]
    daily_df = create_rolling_features(daily_df, roll_targets)

    # 7. 趋势特征
    daily_df = create_trend_features(daily_df)

    # 8. 收益率和Shibor特征
    daily_df = create_yield_shibor_features(daily_df)

    # 9. 处理缺失值
    daily_df = fill_missing_lag_values(daily_df)

    # 10. 用户画像特征
    daily_df = add_user_profile_features(spark, daily_df, data_dir)

    # 11. 整理最终特征集
    final_df = select_final_features(daily_df)

    # 12. 保存
    save_engineered_data(final_df)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"特征工程完成！总耗时: {elapsed:.2f} 秒")
    print(f"{'=' * 60}")

    spark.stop()


if __name__ == "__main__":
    main()