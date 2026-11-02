
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Financial Fraud Detection API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Project_1_Results" / "fraud_detection_pipeline.joblib"

model = joblib.load(MODEL_PATH)


class Transaction(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        data = pd.DataFrame([transaction.features])
        prediction = int(model.predict(data)[0])
        probability = float(model.predict_proba(data)[0][1])

        return {
            "prediction": prediction,
            "result": "Fraud" if prediction == 1 else "Legitimate",
            "fraud_probability": probability
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))
