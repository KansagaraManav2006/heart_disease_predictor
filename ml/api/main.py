"""Internal FastAPI Service for HealthLens AI ML Inference & Monitoring.

Loads trained scikit-learn model & scaler artifacts into memory ONCE at startup,
eliminating per-request process spawning overhead.

Endpoints:
  GET  /internal/v1/health/live
  GET  /internal/v1/health/ready
  GET  /internal/v1/monitoring/drift
  POST /internal/v1/predict/diabetes
  POST /internal/v1/predict/heart
"""

from contextlib import asynccontextmanager
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure ml directory is on sys.path
ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from utils import (
    load_diabetes_model,
    load_diabetes_scaler,
    load_heart_model,
    load_heart_scaler,
    build_diabetes_features,
    build_heart_features,
    predict_diabetes,
    predict_heart,
)
from explain import explain_diabetes, explain_heart
from monitor import compute_dataset_drift

# Global in-memory artifact storage
artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model & scaler artifacts into memory at startup."""
    try:
        artifacts["diabetes_model"] = load_diabetes_model()
        artifacts["diabetes_scaler"] = load_diabetes_scaler()
        artifacts["heart_model"] = load_heart_model()
        artifacts["heart_scaler"] = load_heart_scaler()
        print("[FastAPI ML Service] Loaded all 4 model & scaler artifacts into memory.")
    except Exception as e:
        print(f"[FastAPI ML Service] Failed to load artifacts: {e}")
        raise e
    yield
    artifacts.clear()


app = FastAPI(
    title="HealthLens AI Internal ML Service",
    version="3.0.0",
    lifespan=lifespan,
)


class DiabetesInput(BaseModel):
    age: float = Field(..., gt=0, le=120, description="Age in years")
    gender: str = Field(..., description="Gender (Male, Female, Other)")
    bmi: float = Field(..., gt=10, le=80, description="Body Mass Index")
    glucose: float = Field(..., gt=20, le=600, description="Fasting blood glucose (mg/dL)")
    hba1c: float = Field(..., gt=3, le=20, description="HbA1c level (%)")
    hypertension: Optional[str] = "0"
    heartDisease: Optional[str] = "0"
    smokingHistory: Optional[str] = "never"


class HeartInput(BaseModel):
    age: float = Field(..., gt=0, le=120, description="Age in years")
    gender: str = Field(..., description="Gender (Male, Female)")
    height_cm: float = Field(..., gt=50, le=250, description="Height in cm")
    weight_kg: float = Field(..., gt=20, le=300, description="Weight in kg")
    systolic_bp: float = Field(..., gt=40, le=300, description="Systolic blood pressure (mmHg)")
    diastolic_bp: float = Field(..., gt=30, le=200, description="Diastolic blood pressure (mmHg)")
    cholesterol: float = Field(..., gt=50, le=600, description="Total cholesterol (mg/dL)")
    glucose: float = Field(..., gt=20, le=600, description="Glucose (mg/dL)")
    smoke: Optional[bool] = False
    alco: Optional[bool] = False
    active: Optional[bool] = True


class PredictionResponse(BaseModel):
    condition: str
    prediction: int
    probability: float
    risk_level: str
    risk_band: str
    model_version: str
    explanation: dict


@app.get("/internal/v1/health/live")
def health_live():
    return {"status": "ok"}


@app.get("/internal/v1/health/ready")
def health_ready():
    is_ready = all(
        k in artifacts
        for k in [
            "diabetes_model",
            "diabetes_scaler",
            "heart_model",
            "heart_scaler",
        ]
    )
    if not is_ready:
        raise HTTPException(status_code=503, detail="ML artifacts not loaded")
    return {"status": "ready", "artifacts_loaded": len(artifacts)}


@app.get("/internal/v1/monitoring/drift")
def monitoring_drift():
    # Baseline vs Recent Biometric Distributions
    baseline_samples = {
        "glucose": [90, 95, 100, 105, 110, 115, 120, 125, 130, 140, 150, 160],
        "hba1c": [5.0, 5.2, 5.5, 5.7, 6.0, 6.2, 6.5, 6.8, 7.2, 8.0],
        "systolic_bp": [110, 115, 120, 122, 125, 128, 130, 135, 140, 150],
        "bmi": [20.0, 22.0, 24.0, 25.5, 27.0, 28.5, 30.0, 32.0, 35.0],
    }
    recent_samples = {
        "glucose": [92, 96, 102, 107, 112, 118, 122, 128, 132, 142, 152, 162],
        "hba1c": [5.1, 5.3, 5.6, 5.8, 6.1, 6.3, 6.6, 6.9, 7.3, 8.1],
        "systolic_bp": [112, 116, 121, 124, 126, 130, 132, 137, 142, 152],
        "bmi": [20.2, 22.1, 24.2, 25.7, 27.2, 28.7, 30.2, 32.2, 35.2],
    }

    drift_report = compute_dataset_drift(baseline_samples, recent_samples)
    return drift_report


def get_risk_band(prob: float) -> str:
    if prob < 0.25:
        return "LOW"
    if prob < 0.65:
        return "MODERATE"
    return "HIGH"


@app.post("/internal/v1/predict/diabetes", response_model=PredictionResponse)
def predict_diabetes_endpoint(payload: DiabetesInput):
    try:
        features = build_diabetes_features(
            age=payload.age,
            hypertension_opt="Yes" if str(payload.hypertension) in ("1", "true", "Yes") else "No",
            heart_disease_opt="Yes" if str(payload.heartDisease) in ("1", "true", "Yes") else "No",
            bmi=payload.bmi,
            hba1c=payload.hba1c,
            glucose=payload.glucose,
            gender_opt=payload.gender.capitalize(),
            smoking_opt=payload.smokingHistory or "never",
        )
        pred, prob = predict_diabetes(
            artifacts["diabetes_model"], artifacts["diabetes_scaler"], features
        )
        raw_dict = payload.model_dump()
        explanation = explain_diabetes(
            artifacts["diabetes_model"], artifacts["diabetes_scaler"], features, raw_dict
        )

        risk_band = get_risk_band(prob)

        return {
            "condition": "diabetes",
            "prediction": pred,
            "probability": round(prob, 4),
            "risk_level": "High Risk" if pred == 1 else "Low Risk",
            "risk_band": risk_band,
            "model_version": "diabetes-v3.0",
            "explanation": explanation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/internal/v1/predict/heart", response_model=PredictionResponse)
def predict_heart_endpoint(payload: HeartInput):
    try:
        features, bmi_val = build_heart_features(
            age=payload.age,
            gender=payload.gender.capitalize(),
            height_cm=payload.height_cm,
            weight_kg=payload.weight_kg,
            systolic_bp=payload.systolic_bp,
            diastolic_bp=payload.diastolic_bp,
            cholesterol=payload.cholesterol,
            glucose=payload.glucose,
            smoke=bool(payload.smoke),
            alco=bool(payload.alco),
            active=bool(payload.active),
        )
        pred, prob = predict_heart(
            artifacts["heart_model"], artifacts["heart_scaler"], features
        )
        raw_dict = payload.model_dump()
        explanation = explain_heart(
            artifacts["heart_model"], artifacts["heart_scaler"], features, raw_dict
        )

        risk_band = get_risk_band(prob)

        return {
            "condition": "heart",
            "prediction": pred,
            "probability": round(prob, 4),
            "risk_level": "High Risk" if pred == 1 else "Low Risk",
            "risk_band": risk_band,
            "model_version": "heart-v3.0",
            "explanation": explanation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
