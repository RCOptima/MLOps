import argparse
import logging
import uuid
from abc import ABC, abstractmethod
import mlflow
import mlflow.sklearn
import optuna
import yaml
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.metrics import (accuracy_score, f1_score,
    mean_absolute_error, root_mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target

import trainers
from trainers.trainer import TRAINER_REGISTRY, BaseTrainer

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
 
spark = SparkSession.builder.getOrCreate()
 
 
def infer_task_type(y):
    target_type = type_of_target(y)
    classification_types = {"binary", "multiclass", "multiclass-multioutput", "multilabel-indicator"}
    regression_types = {"continuous", "continuous-multioutput"}
    if target_type in classification_types:
        return "classification"
    elif target_type in regression_types:
        return "regression"
    raise ValueError(f"Could not confidently infer task type from target (detected: '{target_type}')")
 
 
def compute_metrics(task_type, y_true, y_pred):
    metrics = {}
    if task_type == "classification":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        average = "binary" if type_of_target(y_pred) == "binary" else "macro"
        metrics["f1"] = f1_score(y_true, y_pred, average=average)
    else:
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = root_mean_squared_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)
    return metrics
 
 
# def compute_transform_stats(X_before, X_after) -> dict[str, int]:
#     """Cheap, generic before/after stats so the transform step is visible
#     inside every trial run without every DS transform having to implement
#     its own logging -- lets you see, next to the model metrics, whether a
#     bad trial was actually a transform problem (columns dropped, nulls
#     introduced) rather than a model problem."""
#     return {
#         "transform_input_rows": int(len(X_before)),
#         "transform_input_cols": int(X_before.shape[1]),
#         "transform_output_rows": int(len(X_after)),
#         "transform_output_cols": int(X_after.shape[1]),
#         "transform_output_nulls_total": int(X_after.isna().sum().sum()),
#     }
 
 
def run_trial(trainer: BaseTrainer, trial: "optuna.Trial", X_train, y_train,
              X_val, y_val, task_type, primary_metric) -> dict:
    """One Optuna trial == one nested MLflow child run, nested directly
    under the per-model-family run started in train()."""
    with mlflow.start_run(nested=True, run_name=f"{trainer.name}-trial-{trial.number}"):
        mlflow.set_tag("model_family", trainer.name)
        mlflow.set_tag("optuna_trial_number", trial.number)
 
        pipeline = trainer.build_pipeline(trial)
        pipeline.fit(X_train, y_train)
        mlflow.log_params(trial.params)  # whatever trial.suggest_* sampled above
 
        X_train_transformed = pipeline.named_steps["transform"].transform(X_train)
        # mlflow.log_metrics(compute_transform_stats(X_train, X_train_transformed))
 
        y_pred = pipeline.predict(X_val)
        metrics = compute_metrics(task_type, y_val, y_pred)
        mlflow.log_metrics(metrics)
 
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(pipeline, name="model", input_example=X_train.head(5), signature=signature) # config
 
        trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
 
    return metrics[primary_metric]
 
 
def train(train_table: str, val_table: str, model_name: str, cfg: dict):
    pdf_train = spark.table(train_table).toPandas()
    pdf_val = spark.table(val_table).toPandas()

    primary_metric = cfg['primary_metric']
    dbutils.jobs.taskValues.set(key="primary_metric", value=primary_metric)

    target = cfg['target_col']
    dbutils.jobs.taskValues.set(key="target_col", value=target)

    X_train, y_train = pdf_train.drop(columns=[target]), pdf_train[target]
    X_val, y_val = pdf_val.drop(columns=[target]), pdf_val[target]
 
    task_type = infer_task_type(y_train) # could config this?
    logger.info(f"task type '{task_type}' detected")
 
    experiment_path = cfg.get("experiment_path")
    mlflow.set_experiment(experiment_path)
    experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id
    dbutils.jobs.taskValues.set(key="experiment_id", value=experiment_id)
 
    n_trials = cfg.get("n_trials", 20)
    batch_id = str(uuid.uuid4())
    dbutils.jobs.taskValues.set(key="batch_id", value=batch_id)
 
    best_runs = {}
 
    for trainer_name, trainer_cls in TRAINER_REGISTRY.items():
        trainer = trainer_cls()
 
        with mlflow.start_run(run_name=f"{trainer_name}-study"):
            mlflow.set_tag("batch_id", batch_id)
            mlflow.set_tag("model_family", trainer_name)
 
            study = optuna.create_study(direction=trainer.direction)
            study.optimize(
                lambda trial: run_trial(trainer, trial, X_train, y_train, X_val, y_val, task_type, primary_metric),
                n_trials=n_trials,
            )
 
            mlflow.log_params({f"best__{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric(f"best_{primary_metric}", study.best_value)
            mlflow.set_tag("best_child_run_id", study.best_trial.user_attrs["mlflow_run_id"])
 
            best_runs[trainer_name] = study.best_trial.user_attrs["mlflow_run_id"]
            logger.info(
                f"{trainer_name}: best {primary_metric}={study.best_value:.4f} "
                f"(run {best_runs[trainer_name]})"
            )

    #TODO: This only works with one class - there will be more than one best_child_run_id
    # group all of this batch's runs in the UI with: tags.batch_id = '<batch_id>'
    logger.info(f"batch {batch_id} complete: {best_runs}")
    return best_runs

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_table", required=True)
    parser.add_argument("--val_table", required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()
 
    with open("configs/04_train_cfg.yml") as f:
        cfg = yaml.safe_load(f)
 
    train(args.train_table, args.val_table, args.model_name, cfg)

