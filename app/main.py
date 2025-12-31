from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = "final_model/heart_disease_lr_c01.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Heart Disease Prediction API")


class PatientInput(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(input: PatientInput):
    X = np.array(input.features).reshape(1, -1)

    prediction = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0][prediction])

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }
