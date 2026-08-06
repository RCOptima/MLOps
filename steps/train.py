# train.py — today's version is a real, working baseline
import argparse, mlflow
from sklearn.ensemble import RandomForestClassifier

def train(data_path: str, model_name: str, n_estimators: int = 100):
    df = spark.read.table(data_path).toPandas()
    X, y = df.drop("label", axis=1), df["label"]

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X, y)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("train_accuracy", model.score(X, y))
        mlflow.sklearn.log_model(model, "model")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", required=True)
#     parser.add_argument("--model_name", required=True)
#     parser.add_argument("--n_estimators", type=int, default=100)
#     args = parser.parse_args()
#     train(args.data_path, args.model_name, args.n_estimators)