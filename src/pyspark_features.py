"""
Phase 8 - PySpark distributed feature engineering (batch, at scale).

Demonstrates the "PySpark Processing" box of the architecture: the same
entity-level aggregations that features.py computes in pandas, expressed as
distributed Spark DataFrame / Window operations so they scale past a single
machine's memory. Reads the Parquet cache produced in Phase 1, computes
causal per-entity features with Window functions ordered by TransactionDT,
plus card1 frequency encoding, and writes the engineered aggregates back to
Parquet via pandas (avoids the Windows Hadoop write path / winutils).

Run:
  python src/pyspark_features.py

Requires Java 8+ and pyspark<=3.5 (Spark 4 needs Java 17). JAVA_HOME is
already C:\\Java\\jdk-1.8 on this machine.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "DATA" / "cache" / "train_merged.parquet"
OUT = PROJECT_ROOT / "DATA" / "cache" / "spark_entity_features.parquet"

# keep Spark quiet-ish and single-node local
os.environ.setdefault("PYSPARK_PYTHON", "python")


def main() -> None:
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder
        .appName("fraud-feature-engineering")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark {spark.version} session up (local[*]).")

    df = spark.read.parquet(str(CACHE))
    n = df.count()
    print(f"Loaded {n:,} rows from cache.")

    # entity key: card1_addr1
    df = df.withColumn(
        "card1_addr1",
        F.concat_ws("_",
                    F.coalesce(F.col("card1").cast("string"), F.lit("na")),
                    F.coalesce(F.col("addr1").cast("string"), F.lit("na"))),
    )

    # causal window per entity, ordered by arrival time
    w = Window.partitionBy("card1_addr1").orderBy("TransactionDT")
    w_prev = w.rowsBetween(Window.unboundedPreceding, -1)

    df = (
        df
        .withColumn("ent_txn_count_sofar", F.count(F.lit(1)).over(w_prev))
        .withColumn("prev_dt", F.lag("TransactionDT").over(w))
        .withColumn("ent_secs_since_last", F.col("TransactionDT") - F.col("prev_dt"))
        .withColumn("ent_amt_mean_prior", F.avg("TransactionAmt").over(w_prev))
        .withColumn("ent_amt_std_prior", F.stddev("TransactionAmt").over(w_prev))
        .withColumn(
            "ent_amt_z",
            (F.col("TransactionAmt") - F.col("ent_amt_mean_prior"))
            / F.col("ent_amt_std_prior"),
        )
    )

    # card1 frequency encoding (distributed groupBy)
    freq = df.groupBy("card1").agg((F.count(F.lit(1)) / F.lit(n)).alias("card1_freq"))
    df = df.join(freq, on="card1", how="left")

    result = df.select(
        "TransactionID", "isFraud", "card1_addr1",
        "ent_txn_count_sofar", "ent_secs_since_last", "ent_amt_z", "card1_freq",
    )

    # quick sanity: mean of an engineered feature by class (distributed agg)
    print("\nMean ent_txn_count_sofar by class (fraud vs not):")
    (result.groupBy("isFraud")
           .agg(F.avg("ent_txn_count_sofar").alias("avg_velocity"),
                F.avg("card1_freq").alias("avg_card1_freq"))
           .orderBy("isFraud")
           .show())

    # top entities by transaction volume
    print("Busiest entities (card1_addr1) by transaction count:")
    (result.groupBy("card1_addr1")
           .count().orderBy(F.desc("count")).show(5, truncate=False))

    # bring to pandas and write (avoids Windows Hadoop output committer)
    pdf = result.toPandas()
    pdf.to_parquet(OUT, index=False)
    print(f"\nWrote {len(pdf):,} rows of Spark-engineered features -> {OUT.name}")

    spark.stop()


if __name__ == "__main__":
    main()
