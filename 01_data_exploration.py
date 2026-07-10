# -*- coding: utf-8 -*-
"""
============================================================
 01_data_exploration.py
 数据探索分析脚本
 功能：读取四张原始数据表，进行基础统计分析和数据质量检查
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
    col, count, countDistinct, sum as spark_sum, avg, min as spark_min,
    max as spark_max, stddev, desc, asc, when, isnan, isnull,
    to_date, datediff, dayofweek, dayofmonth, month, year, weekofyear
)
from pyspark.sql.types import DoubleType, LongType
import time


def init_spark(app_name="DataExploration"):
    """初始化Spark会话"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_tables(spark, data_dir="Purchase Redemption Data"):
    """读取所有数据表"""
    print("=" * 60)
    print("1. 读取数据表")
    print("=" * 60)

    tables = {}

    # 用户信息表
    tables["user_profile"] = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(f"{data_dir}/user_profile_table.csv")

    # 用户申购赎回数据表（最大的一张表，约280万行）
    tables["user_balance"] = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(f"{data_dir}/user_balance_table.csv")

    # 收益率表
    tables["day_share_interest"] = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(f"{data_dir}/mfd_day_share_interest.csv")

    # Shibor利率表
    tables["bank_shibor"] = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(f"{data_dir}/mfd_bank_shibor.csv")

    for name, df in tables.items():
        print(f"  {name}: {df.count():,} 行, {len(df.columns)} 列")

    return tables


def explore_schema(tables):
    """打印各表Schema"""
    print("\n" + "=" * 60)
    print("2. 数据Schema")
    print("=" * 60)
    for name, df in tables.items():
        print(f"\n--- {name} ---")
        df.printSchema()


def explore_user_profile(df):
    """探索用户信息表"""
    print("\n" + "=" * 60)
    print("3. 用户信息表分析")
    print("=" * 60)

    total_users = df.count()
    print(f"  总用户数: {total_users:,}")

    print("\n  [性别分布]")
    df.groupBy("sex").count() \
        .withColumn("ratio", col("count") / total_users * 100) \
        .orderBy("sex").show()

    print("\n  [城市分布 - Top 10]")
    df.groupBy("city").count() \
        .orderBy(desc("count")) \
        .show(10)

    print("\n  [星座分布]")
    df.groupBy("constellation").count() \
        .withColumn("ratio", col("count") / total_users * 100) \
        .orderBy(desc("count")).show(12)

    print(f"\n  唯一城市数: {df.select('city').distinct().count()}")


def explore_user_balance(df):
    """探索用户申购赎回数据表"""
    print("\n" + "=" * 60)
    print("4. 用户申购赎回数据表分析")
    print("=" * 60)

    total_rows = df.count()
    unique_users = df.select("user_id").distinct().count()
    unique_dates = df.select("report_date").distinct().count()

    print(f"  总行数: {total_rows:,}")
    print(f"  唯一用户数: {unique_users:,}")
    print(f"  唯一日期数: {unique_dates}")

    # 日期范围
    date_range = df.agg(
        spark_min("report_date").alias("min_date"),
        spark_max("report_date").alias("max_date")
    ).collect()[0]
    print(f"  日期范围: {date_range['min_date']} ~ {date_range['max_date']}")

    # 每日用户量和交易量统计
    print("\n  [每日交易统计 - 前5天]")
    daily_stats = df.groupBy("report_date").agg(
        countDistinct("user_id").alias("active_users"),
        spark_sum("total_purchase_amt").alias("daily_total_purchase"),
        spark_sum("total_redeem_amt").alias("daily_total_redeem"),
        spark_sum("direct_purchase_amt").alias("daily_direct_purchase"),
        spark_sum("share_amt").alias("daily_share"),
        spark_sum("consume_amt").alias("daily_consume"),
        spark_sum("transfer_amt").alias("daily_transfer")
    ).orderBy("report_date")
    daily_stats.show(5)

    # 汇总统计
    print("\n  [金额字段汇总统计（单位：分）]")
    amount_cols = ["total_purchase_amt", "total_redeem_amt", "direct_purchase_amt",
                   "share_amt", "consume_amt", "transfer_amt",
                   "purchase_bal_amt", "purchase_bank_amt",
                   "tftobal_amt", "tftocard_amt"]

    for c in amount_cols:
        stats = df.select(
            spark_sum(col(c)).alias("sum"),
            avg(col(c)).alias("avg"),
            stddev(col(c)).alias("std"),
            spark_max(col(c)).alias("max")
        ).collect()[0]
        print(f"    {c}: sum={stats['sum']:,.0f}, avg={stats['avg']:,.2f}, "
              f"std={stats['std']:,.2f}, max={stats['max']:,.0f}")

    # 余额统计
    print("\n  [余额统计]")
    df.select(
        avg("tBalance").alias("avg_balance"),
        stddev("tBalance").alias("std_balance"),
        spark_max("tBalance").alias("max_balance"),
        spark_min("tBalance").alias("min_balance")
    ).show()

    # 零值分析
    print("\n  [关键字段零值比例]")
    for c in ["total_purchase_amt", "total_redeem_amt", "consume_amt",
              "transfer_amt", "share_amt"]:
        zero_count = df.filter(col(c) == 0).count()
        ratio = zero_count / total_rows * 100
        print(f"    {c} = 0: {zero_count:,} ({ratio:.2f}%)")


def explore_yield_and_shibor(yield_df, shibor_df):
    """探索收益率和利率表"""
    print("\n" + "=" * 60)
    print("5. 收益率和Shibor利率表分析")
    print("=" * 60)

    # 收益率表
    print("\n  [收益率表]")
    yield_df.show(5)
    date_range = yield_df.agg(
        spark_min("mfd_date").alias("min"), spark_max("mfd_date").alias("max")
    ).collect()[0]
    print(f"  日期范围: {date_range['min']} ~ {date_range['max']}")

    print("\n  收益率统计:")
    yield_df.select("mfd_daily_yield", "mfd_7daily_yield").describe().show()

    # Shibor利率表
    print("\n  [Shibor利率表]")
    shibor_df.show(5)
    date_range = shibor_df.agg(
        spark_min("mfd_date").alias("min"), spark_max("mfd_date").alias("max")
    ).collect()[0]
    print(f"  日期范围: {date_range['min']} ~ {date_range['max']}")

    print("\n  Shibor各期限利率统计:")
    shibor_df.describe().show()


def check_date_alignment(tables):
    """检查各表日期对齐情况"""
    print("\n" + "=" * 60)
    print("6. 日期对齐检查")
    print("=" * 60)

    # 收集各表日期
    balance_dates = set(
        row["report_date"] for row in
        tables["user_balance"].select("report_date").distinct().collect()
    )
    yield_dates = set(
        row["mfd_date"] for row in
        tables["day_share_interest"].select("mfd_date").distinct().collect()
    )
    shibor_dates = set(
        row["mfd_date"] for row in
        tables["bank_shibor"].select("mfd_date").distinct().collect()
    )

    print(f"  user_balance 日期数: {len(balance_dates)}")
    print(f"  收益率表 日期数: {len(yield_dates)}")
    print(f"  Shibor表 日期数: {len(shibor_dates)}")

    # 交集
    common = balance_dates & yield_dates & shibor_dates
    print(f"  三表共有日期数: {len(common)}")

    # 仅在balance中的日期（收益率表或Shibor表缺失）
    missing_yield = balance_dates - yield_dates
    missing_shibor = balance_dates - shibor_dates
    if missing_yield:
        print(f"  收益率表缺失日期（前10个）: {sorted(missing_yield)[:10]}")
    if missing_shibor:
        print(f"  Shibor表缺失日期（前10个）: {sorted(missing_shibor)[:10]}")


def check_missing_values(tables):
    """检查缺失值"""
    print("\n" + "=" * 60)
    print("7. 缺失值检查")
    print("=" * 60)

    for name, df in tables.items():
        print(f"\n--- {name} ---")
        total = df.count()
        for c in df.columns:
            null_count = df.filter(col(c).isNull()).count()
            if null_count > 0:
                ratio = null_count / total * 100
                print(f"  {c}: NULL={null_count:,} ({ratio:.2f}%)")

    # 特别检查user_balance的category字段（文档说consume_amt=0时为空）
    print("\n  [user_balance category字段空值分析]")
    bal_df = tables["user_balance"]
    cat_cols = ["category1", "category2", "category3", "category4"]
    for c in cat_cols:
        null_count = bal_df.filter(col(c).isNull()).count()
        zero_consume_count = bal_df.filter(
            (col("consume_amt") == 0) & col(c).isNull()
        ).count()
        print(f"    {c}: NULL总数={null_count:,}, consume_amt=0且NULL={zero_consume_count:,}")


def save_exploration_results(tables, output_dir="exploration_output"):
    """保存探索结果（使用Pandas写CSV，避免Hadoop依赖）"""
    print("\n" + "=" * 60)
    print("8. 保存每日聚合结果")
    print("=" * 60)

    import os as _os
    _os.makedirs(output_dir, exist_ok=True)

    # 每日聚合统计（转为Pandas写CSV）
    daily_agg = tables["user_balance"].groupBy("report_date").agg(
        countDistinct("user_id").alias("active_users"),
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
        spark_sum("category1").alias("cat1_total"),
        spark_sum("category2").alias("cat2_total"),
        spark_sum("category3").alias("cat3_total"),
        spark_sum("category4").alias("cat4_total"),
        avg("tBalance").alias("avg_balance"),
        avg("yBalance").alias("avg_yesterday_balance")
    ).orderBy("report_date")

    daily_agg_path = f"{output_dir}/daily_aggregation.csv"
    pdf = daily_agg.toPandas()
    pdf.to_csv(daily_agg_path, index=False, encoding="utf-8-sig")
    print(f"  每日聚合结果已保存至: {daily_agg_path} ({len(pdf)} 行)")

    # 保存各表的描述性统计
    print("\n  探索完成！")


def main():
    spark = init_spark()
    start_time = time.time()

    # 设置数据目录
    data_dir = "Purchase Redemption Data"

    # 读取数据
    tables = read_tables(spark, data_dir)

    # Schema探索
    explore_schema(tables)

    # 各表分析
    explore_user_profile(tables["user_profile"])
    explore_user_balance(tables["user_balance"])
    explore_yield_and_shibor(tables["day_share_interest"], tables["bank_shibor"])

    # 日期对齐
    check_date_alignment(tables)

    # 缺失值
    check_missing_values(tables)

    # 保存结果
    save_exploration_results(tables)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"探索分析完成！总耗时: {elapsed:.2f} 秒")
    print(f"{'=' * 60}")

    spark.stop()


if __name__ == "__main__":
    main()
