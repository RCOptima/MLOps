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

METRIC_FUNCS = {
    "accuracy": accuracy_score,
    "f1_score": f1_score,
    "mse": mean_squared_error,
    "r2": r2_score,
    "rmse": root_mean_squared_error

}

def evaluate(primary_metric, threshold, test_table, batch_id, experiment_id):
    client = MlflowClient()
    greater_is_better = False
    sort_type = 'ASC' if greater_is_better else 'DESC'
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.batch_id = '{batch_id}'",
        order_by=[f"metrics.{primary_metric} {sort_type}"],
    )
    best_run = runs[0]
    run_id = best_run.info.run_id
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model") # config pilot
    # potentiall use this is as insample
    # metrics = mlflow.get_run(run_id).data.metrics
    # score = metrics.get(primary_metric)

    test_pdf = spark.table(test_table).toPandas()
    
    X_test = test_pdf.drop(columns=["target"]) # target should be config
    y_test = test_pdf["target"]
    predictions = model.predict(X_test)

    metric_func = METRIC_FUNCS.get(primary_metric)
    if metric_func is None:
        raise ValueError(f"Unsupported primary_metric: {primary_metric}")
    logger.info(f"choosing {primary_metric} as metric function")

    score = metric_func(y_test, predictions)
    logger.info(f"score - {score}")
    # log the validation score against the same run for traceability
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric(f"test_{primary_metric}", score)

    print(f"{primary_metric} = {score} (threshold {threshold})")
    if score is None or score < threshold:
        sys.exit(1)  # fails the task -> stops the DAG before register

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--primary_metric", default="mse") # should be RMSE?
    p.add_argument("--threshold", type=float, default=0.75) # should there be a threshold or just beat the best model
    p.add_argument("--test_table_silver", required=False)
    p.add_argument("--experiment_id", required=True)
    p.add_argument("--batch_id", required=True)
    args = p.parse_args()
    evaluate(args.primary_metric, args.threshold, args.test_table_silver, args.batch_id, args.experiment_id)