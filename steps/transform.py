
from pyspark.sql import SparkSession
import argparse


def transform(ingest_table, transform_table):
    df = spark.table(ingest_table)

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(str(transform_table))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_table_bronze", required=True)
    parser.add_argument("--train_table_silver", required=True)

    args = parser.parse_args()

    transform(args.train_table_bronze, args.train_table_silver)
