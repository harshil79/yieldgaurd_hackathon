"""FastAPI service wrapping the YieldGuard pass/fail model."""
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

MODEL_PATH = "model.pkl"

app = FastAPI(title="YieldGuard")

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)
model = artifact["model"]
feature_names = artifact["features"]
scaler = artifact["scaler"]
threshold = artifact["threshold"]
fail_idx = list(model.classes_).index(1)
pass_label = [c for c in model.classes_ if c != 1][0]


class PredictRequest(BaseModel):
    features: dict[str, float]

    @field_validator("features")
    @classmethod
    def must_match_expected_features(cls, value):
        missing = set(feature_names) - value.keys()
        if missing:
            raise ValueError(f"missing features: {sorted(missing)}")
        return value


class PredictResponse(BaseModel):
    prediction: int
    fail_probability: float


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    row = pd.DataFrame([[request.features[name] for name in feature_names]], columns=feature_names)
    row_scaled = pd.DataFrame(scaler.transform(row), columns=feature_names)
    fail_probability = float(model.predict_proba(row_scaled)[0][fail_idx])
    # Use the F1-optimal threshold tuned on the validation set during
    # training, not the model's built-in 0.5 default -- see train.py's
    # find_best_threshold().
    prediction = 1 if fail_probability >= threshold else int(pass_label)
    return PredictResponse(prediction=prediction, fail_probability=fail_probability)


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
def ask(request: AskRequest):
    raise HTTPException(status_code=501, detail="Explainer (RAG) not wired yet -- Level 4")
