import argparse
import os
import pandas as pd
import yaml
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main(train_csv: str, target_col: str):
    params = load_params()
    model_cfg = params.get("model", {})
    mtype = model_cfg.get("type", "RandomForestClassifier")

    df = pd.read_csv(train_csv)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if mtype != "RandomForestClassifier":
        raise ValueError("Demo only implements RandomForestClassifier here.")

    clf = RandomForestClassifier(
        n_estimators=model_cfg.get("n_estimators", 200),
        max_depth=model_cfg.get("max_depth", 10),
        random_state=params.get("random_state", 42)
    )

    with mlflow.start_run():
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro")
        prec = precision_score(y, preds, average="macro", zero_division=0)
        rec = recall_score(y, preds, average="macro", zero_division=0)

        mlflow.log_param("model_type", mtype)
        for k, v in model_cfg.items():
            mlflow.log_param(f"model_{k}", v)

        mlflow.log_metric("train_accuracy", acc)
        mlflow.log_metric("train_f1_macro", f1)
        mlflow.log_metric("train_precision_macro", prec)
        mlflow.log_metric("train_recall_macro", rec)

        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, "models/model.pkl")
        mlflow.log_artifact("models/model.pkl")

    print(f"[train] Metrics acc={acc:.3f} f1={f1:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--target", required=True)
    args = ap.parse_args()
    main(args.train, args.target)
