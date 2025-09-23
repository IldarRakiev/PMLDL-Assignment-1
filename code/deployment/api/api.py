from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="California Housing Gradient Boosting API")

MODEL_PATH = "/models/california_gb.joblib"

class PredictRequest(BaseModel):
    features: list

class PredictResponse(BaseModel):
    prediction: float

@app.on_event("startup")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training script first.")
    model_bundle = joblib.load(MODEL_PATH)
    app.state.model = model_bundle["model"]
    app.state.feature_names = model_bundle["feature_names"]
    app.state.target_name = model_bundle["target_name"]
    print("Model loaded:", MODEL_PATH)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "California Housing Gradient Boosting API"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = app.state.model
    x = np.array(req.features, dtype=float).reshape(1, -1)
    pred = float(model.predict(x)[0])
    return PredictResponse(prediction=pred)
