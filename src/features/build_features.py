import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import yaml

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main(input_path: str, output_path: str, sep: str):
    params = load_params()
    target = params.get("target", "quality")
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    df = pd.read_csv(input_path, sep=sep)
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.columns.tolist()

    preproc = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    ct = ColumnTransformer([("num", preproc, num_cols)], remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # For simplicity store a single train file with target included
    train_df = pd.concat([pd.DataFrame(X_train, columns=num_cols).reset_index(drop=True),
                          pd.Series(y_train, name=target).reset_index(drop=True)], axis=1)
    train_df.to_csv(output_path, index=False)

    # Save transformer to reuse in serving
    joblib.dump(ct, "models/transformer.joblib")
    print(f"[features] Saved processed train to {output_path} and transformer to models/transformer.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sep", default=",")
    args = ap.parse_args()
    main(args.input, args.output, args.sep)
