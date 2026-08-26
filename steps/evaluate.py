import argparse, sys, mlflow
from databricks.sdk.runtime import dbutils
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, root_mean_squared_error
import logging
from mlflow import MlflowClient
from train import compute_metrics, infer_task_type

logger = logging.getLogger("evaluate")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

METRICS = {
    "accuracy": (accuracy_score, True),
    "f1_score": (f1_score, True),
    "mse": (mean_squared_error, False),
    "r2": (r2_score, True),
    "rmse": (root_mean_squared_error, False),
}

def register_challenger(run_id, model_name, client, score):
    result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name) # model is config
    client.set_registered_model_alias(model_name, "challenger", result.version)
    logger.info(f'Registered {model_name} as challenger')
    dbutils.jobs.taskValues.set(key="challenger_score", value=score)

def evaluate(primary_metric, threshold, test_table, batch_id, experiment_id, model_name):
    client = MlflowClient()
    metric_func, greater_is_better = METRICS.get(primary_metric, (None, None))
    dbutils.jobs.taskValues.set(key="greater_is_better", value=greater_is_better)

    if metric_func is None:
        raise ValueError(f"Unsupported primary_metric: {primary_metric}")

    if type(greater_is_better) is not bool:
        raise ValueError("greater_is_better is not boolean - greater_is_better - "
                         f"{greater_is_better} which is type {type(greater_is_better)}")

    logger.info(f"choosing {primary_metric} as metric function")

    sort_type = 'DESC' if greater_is_better else 'ASC'
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.batch_id = '{batch_id}'",
        order_by=[f"metrics.{primary_metric} {sort_type}"],
    )
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    dbutils.jobs.taskValues.set(key="best_run_id", value=best_run_id)
    model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/model") # config pilot

    test_pdf = spark.table(test_table).toPandas()
    
    X_test = test_pdf.drop(columns=["target"]) # target should be config
    y_test = test_pdf["target"]
    predictions = model.predict(X_test)

    score = metric_func(y_test, predictions)
    logger.info(f"score - {score}")
    # log the validation score against the same run for traceability
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metric(f"test_{primary_metric}", score)

    def passes_threshold(score, threshold, greater_is_better=True):
        return score >= threshold if greater_is_better else score <= threshold

    if threshold and (score <= threshold if greater_is_better else score >= threshold):
        logger.warning(
            f"Best model select did not reach a threshold of {threshold} for {primary_metric}, "
            "not promoting to challenger and now exiting task")
        sys.exit(1)

    if threshold:
        logger.info(f"{primary_metric} = {score} (threshold {threshold})")

    register_challenger(best_run_id, model_name, client, score)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--primary_metric", default="rmse") # should be RMSE?
    p.add_argument("--threshold", type=float, default=5000) # should there be a threshold or just beat the best model
    p.add_argument("--test_table_silver", required=False)
    p.add_argument("--experiment_id", required=True)
    p.add_argument("--batch_id", required=True)
    p.add_argument("--model_name", required=True)
    args = p.parse_args()
    evaluate(args.primary_metric, args.threshold, args.test_table_silver, args.batch_id, args.experiment_id, args.model_name)
