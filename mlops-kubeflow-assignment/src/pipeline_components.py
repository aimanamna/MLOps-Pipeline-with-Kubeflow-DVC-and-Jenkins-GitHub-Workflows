# src/pipeline_components.py
"""
Kubeflow pipeline components: data extraction (via dvc), preprocessing, training, evaluation.
This file contains Python functions that will be compiled into Kubeflow components (YAML).
"""
import os
import subprocess
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# NOTE: kfp import might fail locally if kfp is not installed; the compile steps use kfp to
# create component YAMLs.
try:
    import kfp
    from kfp.components import create_component_from_func
except Exception:
    kfp = None
    create_component_from_func = None


def run_cmd(cmd, cwd=None, env=None):
    """Utility to run shell commands and raise on failure."""
    print("RUN:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed with code {res.returncode}")


def dvc_get_dataset(dvc_remote_url: str, out_path: str = "/tmp/data/raw_data.csv"):
    """
    Fetch dataset versioned via DVC.
    - dvc_remote_url: path to a repo or remote URL (e.g., a git repo path with DVC tracked file)
      Example usage in CI: dvc get <repo_url> data/raw_data.csv -o /tmp/data/raw_data.csv
    This function uses `dvc get` to retrieve a file tracked by DVC.
    """
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["dvc", "get", dvc_remote_url, "data/raw_data.csv", "-o", str(out_path)]
    run_cmd(cmd)
    return str(out_path)


def preprocess(input_csv: str, output_train: str, output_test: str, test_size: float = 0.2, random_state: int = 42):
    """
    Read raw CSV, do basic cleaning, scaling, and train/test split.
    Saves processed train and test numpy npz files (features + targets).
    """
    df = pd.read_csv(input_csv)
    # simple cleaning: drop NA rows
    df = df.dropna().reset_index(drop=True)

    # For Boston housing, the target column is 'MEDV' or 'target' depending on the source.
    # Try common possibilities:
    if "MEDV" in df.columns:
        target_col = "MEDV"
    elif "target" in df.columns:
        target_col = "target"
    elif "medv" in df.columns:
        target_col = "medv"
    else:
        # If dataset from fetch_openml, the target might be the last column
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(random_state))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    Path(output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(output_test).parent.mkdir(parents=True, exist_ok=True)

    # Save arrays + scaler
    np.savez_compressed(output_train, X=X_train_scaled, y=y_train)
    np.savez_compressed(output_test, X=X_test_scaled, y=y_test)

    # Save scaler to disk for potential serving
    joblib.dump(scaler, str(Path(output_train).with_suffix(".scaler.joblib")))

    result = {
        "train_file": output_train,
        "test_file": output_test,
        "scaler_file": str(Path(output_train).with_suffix(".scaler.joblib")),
    }
    print("Preprocessing result:", result)
    return result


def train_model(train_npz: str, model_out: str, n_estimators: int = 100, random_state: int = 42):
    """
    Train a RandomForestRegressor and save the model as joblib.
    """
    data = np.load(train_npz)
    X = data["X"]
    y = data["y"]

    model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=int(random_state), n_jobs=-1)
    model.fit(X, y)

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Model saved to {model_out}")
    return model_out


def evaluate_model(model_file: str, test_npz: str, metrics_out: str):
    """
    Load model and test data, compute metrics (MSE, RMSE, R2) and write them to JSON.
    """
    model = joblib.load(model_file)
    data = np.load(test_npz)
    X_test = data["X"]
    y_test = data["y"]

    preds = model.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, preds))

    metrics = {"mse": mse, "rmse": rmse, "r2": r2}
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f)

    print("Evaluation metrics:", metrics)
    return metrics_out


# === optional: compile helpers (run locally to create component yaml files) ===
if __name__ == "__main__" and create_component_from_func:
    # compile components into components/ directory
    out_dir = Path("components")
    out_dir.mkdir(exist_ok=True)
    print("Compiling components to YAML in", out_dir)

    create_component_from_func(dvc_get_dataset, output_component_file=str(out_dir / "dvc_get_dataset.yaml"))
    create_component_from_func(preprocess, output_component_file=str(out_dir / "preprocess.yaml"))
    create_component_from_func(train_model, output_component_file=str(out_dir / "train_model.yaml"))
    create_component_from_func(evaluate_model, output_component_file=str(out_dir / "evaluate_model.yaml"))
    print("Compiled component YAMLs.")
