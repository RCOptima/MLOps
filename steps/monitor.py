import argparse
import sys
import logging
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    AiGatewayUsageTrackingConfig,
)
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType
from databricks.sdk.errors import NotFound
from pyspark.sql import SparkSession
 
from deploy import endpoint_name_for
 
logger = logging.getLogger("monitor")
logger.setLevel(logging.INFO)
 
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
 
 
def seed_baseline_traffic(endpoint_name, val_df, label_col, n=10):
    w = WorkspaceClient()
    sample = val_df.sample(n=min(n, len(val_df)))
    records = sample.drop(columns=[label_col]).to_dict(orient="records")
    w.serving_endpoints.query(name=endpoint_name, dataframe_records=records)
 
 
def setup_monitoring(model_name, catalog_name, schema_name, val_table, endpoint_name=None):
    model_name_prefix = "capstone"  # config
    spark = SparkSession.builder.getOrCreate()
    w = WorkspaceClient()
 
    endpoint_name = endpoint_name or endpoint_name_for(model_name)
    try:
        w.serving_endpoints.put_ai_gateway(
            name=endpoint_name,
            inference_table_config=AiGatewayInferenceTableConfig(
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name_prefix=model_name_prefix, # test
                enabled=True,
            ),
            usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        )
    except Exception as e:
        logger.error(f"failed to enable AI Gateway inference logging: {e}")
        sys.exit(1)
 
    label_col = "target"
 
    val_pdf = spark.table(val_table).toPandas()
    seed_baseline_traffic(endpoint_name, val_pdf, label_col)
    payload_table = f"{catalog_name}.{schema_name}.{model_name_prefix}_payload"
    parsed_view = f"{catalog_name}.{schema_name}.{model_name_prefix}_payload_parsed"
    
    existing_monitor = w.quality_monitors.get(table_name=parsed_view)
    try:
        spark.sql(f"""
            CREATE VIEW IF NOT EXISTS {parsed_view} AS
            SELECT
                p.databricks_request_id AS request_id,
                p.request_time          AS timestamp,
                p.served_entity_id      AS model_id,
                CAST(get_json_object(p.response, '$.predictions[0]') AS DOUBLE) AS prediction
            FROM {payload_table} p
        """)
    except Exception as e:
        logger.error(f"failed to create parsed payload view: {e}")
        sys.exit(1)
    try:
        existing_monitor = w.quality_monitors.get(table_name=parsed_view)
    except NotFound:
        existing_monitor = None
    try:
        monitor_kwargs = {
            "table_name": parsed_view,
            "output_schema_name": f"{catalog_name}.{schema_name}",
            "inference_log": MonitorInferenceLog(
                problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
                prediction_col="prediction",
                timestamp_col="timestamp",
                model_id_col="model_id",
                granularities=["1 day"],
            ),
        }
        assets_dir = (f"/Workspace/Users/ross.campbell@optimapartners.co.uk/"
                      f"MLOps_framework/{model_name}")

        if existing_monitor:
            logger.info(f"quality monitor already exists on '{parsed_view}', updating config")
            w.quality_monitors.update(**monitor_kwargs)
        else:
            logger.info(f"creating new quality monitor on '{parsed_view}'")
            w.quality_monitors.create(assets_dir=assets_dir, **monitor_kwargs)

    except Exception as e:
        logger.error(f"failed to create quality monitor: {e}")
        sys.exit(1)
             # --   se.entity_version                                        AS model_version
     # assumes no batched sending requests
        # JOIN system.serving.served_entities se # addnthis if i get permissions for schema
        #   ON p.served_entity_id = se.served_entity_id
        # WHERE p.status_code = 200
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--val_table_silver", required=True)
    # p.add_argument("--endpoint_name", required=False, default=None)
    args = p.parse_args()
    setup_monitoring(args.model_name, args.catalog, args.schema, args.val_table_silver) # , args.endpoint_name)
 
