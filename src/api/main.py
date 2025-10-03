from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from . import __init__  # noqa
from src.models.infer_utils import load_transformer, load_latest_model

app = FastAPI(title="MLOps Internship API", version="0.1.0")

transformer = None
model = None

class PredictRequest(BaseModel):
    features: list  # list of lists (batch) or list (single)

@app.on_event("startup")
def load_artifacts():
    global transformer, model
    transformer = load_transformer("models/transformer.joblib")
    model = load_latest_model("models/model.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.features, dtype=float)
    X_t = transformer.transform(X)
    preds = model.predict(X_t)
    return {"predictions": preds.tolist()}
