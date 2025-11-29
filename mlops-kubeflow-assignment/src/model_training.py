# src/model_training.py
"""
Standalone training script that mirrors the component train step.
Useful for local testing (python src/model_training.py --train /path/to/train.npz --out model.joblib)
"""
import argparse
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

def main(args):
    data = np.load(args.train)
    X = data["X"]
    y = data["y"]
    model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1)
    model.fit(X, y)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print("Saved model to", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)
