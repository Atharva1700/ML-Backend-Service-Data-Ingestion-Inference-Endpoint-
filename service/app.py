from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .inference import get_predictions
from .evaluation import run_evaluation

app = FastAPI()

class PredictionRequest(BaseModel):
    features: List[float]

class EvalRequest(BaseModel):
    predictions: List[List[float]]
    ground_truth: List[int]

@app.post("/predict")
async def predict(req: PredictionRequest) -> dict:
    preds = get_predictions(req.features)
    return {'predictions': preds}

@app.post("/evaluate")
async def evaluate(req: EvalRequest) -> dict:
    results = run_evaluation(req.predictions, req.ground_truth)
    return results

@app.get("/health")
async def health():
    return {"status": "ok"}
