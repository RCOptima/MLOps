import argparse
import mlflow
from mlflow import MlflowClient
import logging
logger = logging.getLogger("register")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def register(primary_metric, model_name, challenger_score, greater_is_better):
    client = MlflowClient()
    challenger = client.get_model_version_by_alias(model_name, "challenger")

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_score = float(client.get_run(champion.run_id).data.metrics[primary_metric])
        logger.info(f'Comparing current champion model that with a {primary_metric} ' 
                    f'{champion_score:4f} and challenger {primary_metric} {challenger_score:4f}')

    except mlflow.exceptions.RestException:
        logger.warning('Cannot detect champion model, auto promoting challenger to champion')
        champion_score = None

    if champion_score is None or (
        challenger_score > champion_score if greater_is_better else challenger_score < champion_score):
        logger.info('Promoting challenger to champion')
        client.set_registered_model_alias(model_name, "champion", challenger.version)

    if champion_score is None or challenger_score > champion_score:
        logger.warning
        client.set_registered_model_alias(model_name, "champion", challenger.version)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--primary_metric", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--challenger_score", type=float, required=True)
    p.add_argument("--greater_is_better", type=bool, required=True)
    args = p.parse_args()
    register(args.primary_metric, args.model_name, args.challenger_score, args.greater_is_better)
