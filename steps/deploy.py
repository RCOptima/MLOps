import argparse
from mlflow.deployments import get_deploy_client
from mlflow import MlflowClient
import sys
import logging
import time

logger = logging.getLogger("deploy")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def endpoint_name_for(model_name):
    """Shared naming convention so deploy.py and monitor.py always agree on the endpoint name."""
    return f"{model_name.replace('.', '-')}-endpoint"


def build_config(model_name, champion_version, catalog_name, schema_name):
    return {
        "served_entities": [{
            "entity_name": model_name,
            "entity_version": champion_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }],
    }

def wait_for_ready(deploy_client, endpoint_name, timeout_s=600, poll_interval_s=15):
    """Poll until the endpoint is READY, or raise on timeout/failure."""
    start = time.time()
    while True:
        endpoint = deploy_client.get_endpoint(endpoint_name)
        state = endpoint.get("state", {})
        ready = state.get("ready")
        config_update = state.get("config_update")
 
        logger.info(f"  endpoint state: ready={ready}, config_update={config_update}")
 
        if ready == "READY" and config_update in (None, "NOT_UPDATING"):
            return endpoint
        if config_update == "UPDATE_FAILED":
            raise RuntimeError(f"Endpoint '{endpoint_name}' update failed: {state}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Endpoint '{endpoint_name}' did not become ready within {timeout_s}s")
        time.sleep(poll_interval_s)

def deploy(model_name, catalog_name, schema_name):
    mlflow_client = MlflowClient()
    deploy_client = get_deploy_client("databricks")
    champion_version = mlflow_client.get_model_version_by_alias(model_name, "champion").version
    endpoint_name = endpoint_name_for(model_name)
    config = build_config(model_name, champion_version, catalog_name, schema_name)

    try:
        existing = deploy_client.get_endpoint(endpoint_name)
    except Exception:
        existing = None

    try:
        if existing:
            logger.info(f"endpoint '{endpoint_name}' exists, updating with current champion")
            deploy_client.update_endpoint_config(endpoint=endpoint_name, config=config)
        else:
            logger.info(f"creating new endpoint '{endpoint_name}'")
            deploy_client.create_endpoint(name=endpoint_name, config=config)
    except Exception as e:
        logger.error(f"deployment call failed: {e}")
        sys.exit(1)

    try:
        wait_for_ready(deploy_client, endpoint_name)
    except (RuntimeError, TimeoutError) as e:
        logger.error(f"deployment did not succeed: {e}")
        sys.exit(1)

    logger.info(f"endpoint '{endpoint_name}' is ready (champion version {champion_version})")
    return endpoint_name


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    args = p.parse_args()
    deploy(args.model_name, args.catalog, args.schema)