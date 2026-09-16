from pyspark.sql import DataFrame, SparkSession
import yaml
import argparse
import logging

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def _read_delta_table(spark: SparkSession, source_cfg: dict) -> DataFrame:
    return spark.read.format("delta").load(source_cfg["path"])
 
 
def _read_uc_table(spark: SparkSession, source_cfg: dict) -> DataFrame:
    # e.g. "catalog.schema.table_name" — Unity Catalog managed table
    return spark.table(source_cfg["path"])
 
 
def _read_parquet(spark: SparkSession, source_cfg: dict) -> DataFrame:
    return spark.read.parquet(source_cfg["path"])
 

def _read_csv(spark: SparkSession, source_cfg: dict) -> DataFrame:
    options = source_cfg.get("options", {"header": True, "inferSchema": True})
    return spark.read.options(**options).csv(source_cfg["path"])
 
 
READER_REGISTRY = {
    "delta_table": _read_delta_table,
    "uc_table": _read_uc_table,
    "parquet": _read_parquet,
    "csv": _read_csv,
}

def ingest(catalog, schema, output_path, spark, cfg):
    source_cfg = cfg["source"]
    source_type = source_cfg["type"]
    logger.debug(f"source type {source_type} - detected with path {source_cfg["path"]}")
    if source_type not in READER_REGISTRY:
        raise ValueError(
            f"Unsupported source type '{source_type}'. "
            f"Supported: {list(READER_REGISTRY.keys())}. "
            f"To add a new one, add a reader function to READER_REGISTRY in ingest.py."
        )
    df = READER_REGISTRY[source_type](spark, source_cfg)

    spark.sql(f"""
        CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}
    """)

    df.write.format("delta").mode("overwrite").saveAsTable(str(output_path))
    logger.info(f"Ingest successful, delta table stored at {output_path}")

    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--ingested_data_path", required=True)
    args = parser.parse_args()

    with open("configs/01_ingest_cfg.yml") as f:
        cfg = yaml.safe_load(f)
    # do we need this?
    spark = SparkSession.builder.appName(f"ingest-{cfg['dataset_name']}").getOrCreate()
    ingest(args.catalog, args.schema, args.ingested_data_path, spark, cfg)