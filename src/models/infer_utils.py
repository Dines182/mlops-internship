import joblib
import mlflow.pyfunc

def load_transformer(path: str = "models/transformer.joblib"):
    return joblib.load(path)

def load_latest_model(model_path: str = "models/model.pkl"):
    # For simplicity, load from local file. You can change to mlflow model URI.
    return joblib.load(model_path)
