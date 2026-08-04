# evaluate.py
import argparse, sys, mlflow

def evaluate(run_id, primary_metric, threshold):
    metrics = mlflow.get_run(run_id).data.metrics
    score = metrics.get(primary_metric)
    print(f"{primary_metric} = {score} (threshold {threshold})")
    if score is None or score < threshold:
        sys.exit(1)  # fails the task -> stops the DAG before register

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--primary_metric", default="f1_score")
    p.add_argument("--threshold", type=float, default=0.75)
    args = p.parse_args()
    evaluate(args.run_id, args.primary_metric, args.threshold)