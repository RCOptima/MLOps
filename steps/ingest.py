from pyspark.sql import SparkSession
import argparse


def ingest(raw_data, catalog, schema, path):
    spark = SparkSession.builder.getOrCreate()

    spark.sql(f"""
        CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}
    """)

    df = spark.table(raw_data)

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(str(path))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--ingested_data_path", required=True)
    parser.add_argument("--raw_data", required=True)
    args = parser.parse_args()

    ingest(args.raw_data, args.catalog, args.schema, args.ingested_data_path)