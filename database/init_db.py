# -*- coding: utf-8 -*-
"""
初始化 SQLite 数据库：建表 + 导入 CSV + 数据质量校验

用法:
    python database/init_db.py
    python database/init_db.py --db data/purchase_redemption.db --force
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "purchase_redemption.db"
SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"
RAW_DIR = ROOT / "Purchase Redemption Data"
EXPLORATION_DIR = ROOT / "exploration_output"
FEATURE_DIR = ROOT / "feature_engineering_output"
MODELING_DIR = ROOT / "modeling_output"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def run_schema(conn: sqlite3.Connection) -> None:
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def import_user_profile(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(RAW_DIR / "user_profile_table.csv")
    df.to_sql("user_profile", conn, if_exists="replace", index=False)
    return len(df)


def import_user_balance(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(RAW_DIR / "user_balance_table.csv")
    rename = {
        "tBalance": "t_balance",
        "yBalance": "y_balance",
    }
    df = df.rename(columns=rename)
    df.to_sql("user_balance", conn, if_exists="replace", index=False)
    return len(df)


def import_yield(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(RAW_DIR / "mfd_day_share_interest.csv")
    df.to_sql("mfd_day_share_interest", conn, if_exists="replace", index=False)
    return len(df)


def import_shibor(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(RAW_DIR / "mfd_bank_shibor.csv")
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    df.to_sql("mfd_bank_shibor", conn, if_exists="replace", index=False)
    return len(df)


def import_predict_template(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(
        RAW_DIR / "comp_predict_table.csv",
        header=None,
        names=["report_date", "purchase", "redeem"],
    )
    df.to_sql("comp_predict_template", conn, if_exists="replace", index=False)
    return len(df)


def import_daily_summary(conn: sqlite3.Connection) -> int:
    path = EXPLORATION_DIR / "daily_aggregation.csv"
    if not path.exists():
        print(f"  [跳过] daily_summary: 未找到 {path}")
        return 0
    df = pd.read_csv(path)
    df.to_sql("daily_summary", conn, if_exists="replace", index=False)
    return len(df)


def import_daily_features(conn: sqlite3.Connection) -> int:
    path = FEATURE_DIR / "engineered_features.csv"
    if not path.exists():
        print(f"  [跳过] daily_features: 未找到 {path}")
        return 0
    df = pd.read_csv(path)
    df.to_sql("daily_features", conn, if_exists="replace", index=False)
    return len(df)


def import_ml_datasets(conn: sqlite3.Connection) -> tuple[int, int]:
    train_path = MODELING_DIR / "train_data.csv"
    val_path = MODELING_DIR / "val_data.csv"
    train_n = val_n = 0
    if train_path.exists():
        df = pd.read_csv(train_path)
        df.to_sql("ml_train_data", conn, if_exists="replace", index=False)
        train_n = len(df)
    else:
        print(f"  [跳过] ml_train_data: 未找到 {train_path}")
    if val_path.exists():
        df = pd.read_csv(val_path)
        df.to_sql("ml_val_data", conn, if_exists="replace", index=False)
        val_n = len(df)
    else:
        print(f"  [跳过] ml_val_data: 未找到 {val_path}")
    return train_n, val_n


def seed_dataset_split(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM dataset_split")
    conn.execute(
        """
        INSERT INTO dataset_split (report_date, split_type, note)
        SELECT report_date, 'train', '2013-07-01 ~ 2014-07-31'
        FROM ml_train_data
        """
    )
    conn.execute(
        """
        INSERT INTO dataset_split (report_date, split_type, note)
        SELECT report_date, 'val', '2014-08-01 ~ 2014-08-31'
        FROM ml_val_data
        """
    )
    predict_dates = pd.read_csv(
        RAW_DIR / "comp_predict_table.csv",
        header=None,
        names=["report_date", "purchase", "redeem"],
    )["report_date"]
    conn.executemany(
        """
        INSERT INTO dataset_split (report_date, split_type, note)
        VALUES (?, 'predict', '2014-09 预测目标日')
        """,
        [(int(d),) for d in predict_dates],
    )
    conn.commit()


def run_quality_checks(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM data_quality_report")

    violation_count = conn.execute(
        "SELECT COUNT(*) FROM v_balance_violations"
    ).fetchone()[0]
    total_rows = conn.execute("SELECT COUNT(*) FROM user_balance").fetchone()[0]
    conn.execute(
        """
        INSERT INTO data_quality_report (check_name, check_result, metric_value, detail)
        VALUES (?, ?, ?, ?)
        """,
        (
            "balance_equation",
            "pass" if violation_count == 0 else "fail",
            violation_count,
            f"违反「今日余额=昨日余额+申购-赎回」的记录数: {violation_count}/{total_rows}",
        ),
    )

    date_gap = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT ds.report_date
            FROM daily_summary ds
            LEFT JOIN mfd_day_share_interest y ON ds.report_date = y.mfd_date
            WHERE y.mfd_date IS NULL
        )
        """
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO data_quality_report (check_name, check_result, metric_value, detail)
        VALUES (?, ?, ?, ?)
        """,
        (
            "yield_date_alignment",
            "pass" if date_gap == 0 else "warn",
            date_gap,
            f"daily_summary 与收益率表未对齐日期数: {date_gap}",
        ),
    )

    null_consume_category = conn.execute(
        """
        SELECT COUNT(*) FROM user_balance
        WHERE consume_amt = 0
          AND (category1 IS NOT NULL OR category2 IS NOT NULL
               OR category3 IS NOT NULL OR category4 IS NOT NULL)
        """
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO data_quality_report (check_name, check_result, metric_value, detail)
        VALUES (?, ?, ?, ?)
        """,
        (
            "consume_category_consistency",
            "pass" if null_consume_category == 0 else "warn",
            null_consume_category,
            "consume_amt=0 时 category 字段应为空",
        ),
    )
    conn.commit()


def print_summary(conn: sqlite3.Connection, db_path: Path) -> None:
    tables = [
        "user_profile",
        "user_balance",
        "mfd_day_share_interest",
        "mfd_bank_shibor",
        "comp_predict_template",
        "daily_summary",
        "daily_features",
        "ml_train_data",
        "ml_val_data",
        "dataset_split",
    ]
    print("\n" + "=" * 60)
    print(f"数据库: {db_path}")
    print("=" * 60)
    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:28s} {count:>10,} 行")
        except sqlite3.OperationalError:
            print(f"  {table:28s} {'(未导入)':>10}")

    print("\n数据质量检查:")
    for row in conn.execute(
        "SELECT check_name, check_result, detail FROM data_quality_report ORDER BY check_id"
    ):
        print(f"  [{row[1]:>4}] {row[0]}: {row[2]}")
    print("=" * 60)


def init_database(db_path: Path, force: bool = False) -> None:
    if db_path.exists():
        if force:
            db_path.unlink()
            print(f"已删除旧数据库: {db_path}")
        else:
            print(f"数据库已存在: {db_path}，使用 --force 可重建")

    conn = connect(db_path)
    try:
        print("1. 执行 schema.sql ...")
        run_schema(conn)

        print("2. 导入原始数据 ...")
        print(f"   user_profile          {import_user_profile(conn):>10,} 行")
        print(f"   user_balance          {import_user_balance(conn):>10,} 行")
        print(f"   mfd_day_share_interest {import_yield(conn):>10,} 行")
        print(f"   mfd_bank_shibor       {import_shibor(conn):>10,} 行")
        print(f"   comp_predict_template {import_predict_template(conn):>10,} 行")

        print("3. 导入加工数据 ...")
        print(f"   daily_summary         {import_daily_summary(conn):>10,} 行")
        print(f"   daily_features        {import_daily_features(conn):>10,} 行")
        train_n, val_n = import_ml_datasets(conn)
        print(f"   ml_train_data         {train_n:>10,} 行")
        print(f"   ml_val_data           {val_n:>10,} 行")

        if train_n and val_n:
            print("4. 写入 dataset_split ...")
            seed_dataset_split(conn)

        print("5. 运行数据质量检查 ...")
        run_quality_checks(conn)

        print_summary(conn, db_path)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化资金流入流出预测数据库")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite 文件路径")
    parser.add_argument("--force", action="store_true", help="重建数据库")
    args = parser.parse_args()
    init_database(args.db.resolve(), force=args.force)


if __name__ == "__main__":
    main()
