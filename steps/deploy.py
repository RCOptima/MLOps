import argparse
from mlflow.deployments import get_deploy_client
from mlflow import MlflowClient
import sys
import logging

logger = logging.getLogger("deploy")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def build_config(model_name, champion_version, catalog_name, schema_name):
    return {
        "served_entities": [{
            "entity_name": model_name,
            "entity_version": champion_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }],
    }

def deploy(model_name, catalog_name, schema_name):
    mlflow_client = MlflowClient()
    deploy_client = get_deploy_client("databricks")
    champion_version = mlflow_client.get_model_version_by_alias(model_name, "champion").version
    endpoint_name = f"{model_name.replace('.', '-')}-endpoint"
    config = build_config(model_name, champion_version, catalog_name, schema_name)

    try:
        existing = deploy_client.get_endpoint(endpoint_name)
    except Exception:
        existing = None

    try:
        if existing:
            logger.info(f"endpoint '{endpoint_name}' exists, updating config")
            deploy_client.update_endpoint_config(endpoint=endpoint_name, config=config)
        else:
            logger.info(f"creating new endpoint '{endpoint_name}'")
            deploy_client.create_endpoint(name=endpoint_name, config=config)
    except Exception as e:
        logger.error(f"deployment call failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    args = p.parse_args()
    deploy(args.model_name, args.catalog, args.schema)
