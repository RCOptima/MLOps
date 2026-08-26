import argparse
import mlflow
from mlflow import MlflowClient
from evaluate import METRICS

def register(primary_metric, model_name, challenger_score, greater_is_better):
    client = MlflowClient()
    try:
        challenger = client.get_model_version_by_alias(model_name, "challenger")
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_score = float(client.get_run(champion.run_id).data.metrics[primary_metric])
    except mlflow.exceptions.RestException:
        champion_score = None
    if greater_is_better:
        if champion_score is None or (
            challenger_score > champion_score if greater_is_better else challenger_score < champion_score):
            client.set_registered_model_alias(model_name, "champion", challenger.version)
    else:
        if champion_score is None or challenger_score > champion_score:
            client.set_registered_model_alias(model_name, "champion", challenger.version)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--primary_metric", default="rmse") # should be RMSE?
    p.add_argument("--model_name", required=True)
    p.add_argument("--challenger_score", type=float, required=True)
    p.add_argument("--greater_is_better", type=bool, required=True)
    args = p.parse_args()
    register(args.primary_metric, args.model_name, args.challenger_score, args.greater_is_better)
