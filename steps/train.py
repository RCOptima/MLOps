import argparse

import mlflow
import mlflow.sklearn
import uuid
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, root_mean_squared_error, r2_score,
)
from pyspark.sql import SparkSession
from itertools import product
import logging

logger = logging.getLogger("split")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
spark = SparkSession.builder.getOrCreate()

param_grid = { # config - or perahps import this? #TODO: other forms of gridsearch i.e. hyperopt etc//
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
}

def generate_grid(param_grid):
    if not param_grid:
        yield {}
        return

    keys = param_grid.keys()
    values = param_grid.values()

    for combination in product(*values):
        yield dict(zip(keys, combination))

def model_factory(params):
    return RandomForestRegressor(
        random_state=42,
        **params,
    )

def infer_task_type(y):
    target_type = type_of_target(y)
    classification_types = {"binary", "multiclass", "multiclass-multioutput", "multilabel-indicator"}
    regression_types = {"continuous", "continuous-multioutput"}

    if target_type in classification_types:
        return "classification"
    elif target_type in regression_types:
        return "regression"
    else:
        raise ValueError(f"Could not confidently infer task type from target (detected: '{target_type}')")


def compute_metrics(task_type, model, X_val, y_val):
    y_pred = model.predict(X_val)
    metrics = {}

    if task_type == "classification":
        metrics["accuracy"] = accuracy_score(y_val, y_pred)
        average = "binary" if type_of_target(y_val) == "binary" else "macro"
        metrics["f1"] = f1_score(y_val, y_pred, average=average)
        # roc_auc needs predict_proba and only makes sense for binary/multiclass with probabilities
        if hasattr(model, "predict_proba"):
            try:
                if type_of_target(y_val) == "binary":
                    metrics["roc_auc"] = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(y_val, model.predict_proba(X_val), multi_class="ovr")
            except ValueError:
                pass  # skip if it can't be computed (e.g. class missing from validation set)
    else:  # regression
        metrics["mae"] = mean_absolute_error(y_val, y_pred)
        metrics["rmse"] = root_mean_squared_error(y_val, y_pred)
        metrics["r2"] = r2_score(y_val, y_pred)

    return metrics

def train(silver_train_table: str, silver_val_table: str, model_name: str, validate: bool):
    # remove silver_val_table as required
    pdf_train = spark.table(silver_train_table).toPandas()

    X_train = pdf_train.drop(columns=["target"])
    y_train = pdf_train["target"]

    if validate:
        pdf_val = spark.table(silver_val_table).toPandas()
        X_val = pdf_val.drop(columns=["target"])
        y_val = pdf_val["target"]

    task_type = infer_task_type(y_train) # should ge this also from input
    logger.info(f"task type {task_type} detected")

    experiment_path = f"/Workspace/Users/ross.campbell@optimapartners.co.uk/pilot" # should be config
    mlflow.set_experiment(experiment_path)
    experiment = mlflow.get_experiment_by_name(experiment_path)
    experiment_id = experiment.experiment_id
    dbutils.jobs.taskValues.set(key="experiment_id", value=experiment_id)
    results = []

    batch_id = str(uuid.uuid4())
    dbutils.jobs.taskValues.set(key="batch_id", value=batch_id)
    for i, params in enumerate(generate_grid(param_grid)):

        with mlflow.start_run():
            mlflow.set_tag("batch_id", batch_id)
            model = model_factory(params)

            model.fit(X_train, y_train)

            metrics = compute_metrics(
                task_type,
                model,
                X_val,
                y_val,
            )

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                name=f"model", # config?
                input_example=X_train
            )
            # result = {
            #     "params": params,
            #     "model": model,
            #     "metrics": metrics,
            #     "run_id": mlflow.active_run().info.run_id
            # }
            # logger.info(result)
            # results.append(result)

    # optimisation_metric = 'rmse' # config this
    # greater_is_better = False # and this

    # if greater_is_better:
    #     best = max(results, key=lambda x: x["metrics"][optimisation_metric])
    # else:
    #     best = min(results,key=lambda x: x["metrics"][optimisation_metric])

    # best_run_id = best['run_id']
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_table_silver", required=True)
    parser.add_argument("--val_table_silver", required=False)
    parser.add_argument("--validate", required=True) # does this need to be true?

    parser.add_argument("--model_name", required=True)

    args = parser.parse_args()

    train(args.train_table_silver, args.val_table_silver, args.model_name, args.validate)