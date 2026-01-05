from fastapi import FastAPI, HTTPException
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


EXPECTED_FEATURES = model.named_steps["clf"].coef_.shape[1]

@app.post("/predict")
def predict(input: PatientInput):
    if len(input.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features, got {len(input.features)}"
        )

    X = np.array(input.features).reshape(1, -1)

    prediction = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0][prediction])

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }
