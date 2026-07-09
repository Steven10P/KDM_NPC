"""
Helper script: export MLflow run data to experiments/results/
Run from the project root: python scripts/extract_mlflow_results.py
"""
import os
import json
import pandas as pd

def extract_mlflow_to_csv(tracking_uri="sqlite:///mlflow.db",
                           experiment_name="kdm-pnc-mnist-noise",
                           out_dir="experiments/results"):
    try:
        import mlflow
    except ImportError:
        print("[ERROR] mlflow not installed. Run: pip install mlflow")
        return None

    os.makedirs(out_dir, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Find the experiment
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"[WARN] Experiment '{experiment_name}' not found. Listing all experiments:")
        for e in client.search_experiments():
            print(f"  - {e.name} (id={e.experiment_id})")
        return None

    runs = client.search_runs([exp.experiment_id])
    records = []
    for run in runs:
        m = run.data.metrics
        p = run.data.params
        records.append({
            "run_id": run.info.run_id,
            "model": p.get("model", "unknown"),
            "dataset": p.get("dataset", ""),
            "noise_sigma": p.get("noise_sigma", ""),
            "pca_dims": p.get("pca_dims", ""),
            "epochs": p.get("epochs", ""),
            "train_acc": m.get("train_acc"),
            "test_acc": m.get("test_acc"),
            "train_loss": m.get("train_loss"),
            "test_loss": m.get("test_loss"),
            "status": run.info.status,
        })

    df = pd.DataFrame(records)
    out_path = os.path.join(out_dir, "mlflow_results.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(df)} runs to {out_path}")
    return df


if __name__ == "__main__":
    df = extract_mlflow_to_csv()
    if df is not None:
        print(df.to_string())
