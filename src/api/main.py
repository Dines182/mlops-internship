from typing import List, Annotated
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from src.models.infer_utils import load_transformer, load_latest_model

app = FastAPI(title="MLOps Internship API", version="0.1.0")

# ---- Request model (simple; no examples on the field/model) ----
class PredictRequest(BaseModel):
    features: List[List[float]]

# Lazy-load artifacts so /docs works even if files aren't present yet
_transformer = None
_model = None
def ensure_artifacts():
    global _transformer, _model
    if _transformer is None:
        _transformer = load_transformer("models/transformer.joblib")
    if _model is None:
        _model = load_latest_model("models/model.pkl")

@app.get("/")
def home():
    return {
        "message": "MLOps Internship API running",
        "docs": "/docs",
        "health": "/health",
        "predict_example": {
            "features": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
        },
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # sanity check: must be 11 features per row
    if any(len(row) != 11 for row in req.features):
        raise HTTPException(
            status_code=400,
            detail="Each feature row must have exactly 11 numeric values."
        )

    ensure_artifacts()
    X = np.array(req.features, dtype=float)

    # Always predict
    preds = _model.predict(X)

    # Try to also return probabilities (if model supports it)
    probs = None
    try:
        probs = _model.predict_proba(X).tolist()
    except Exception:
        # Some models don't have predict_proba
        pass

    return {
        "predictions": preds.tolist(),
        "probabilities": probs
    }
