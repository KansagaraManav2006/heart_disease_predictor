"""Internal FastAPI Service for HealthLens AI ML Inference.

Loads trained scikit-learn model & scaler artifacts into memory ONCE at startup,
eliminating per-request process spawning overhead.

Endpoints:
  GET  /internal/v1/health/live
  GET  /internal/v1/health/ready
  POST /internal/v1/predict/diabetes
  POST /internal/v1/predict/heart
"""

from contextlib import asynccontextmanager
import sys
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Ensure ml directory is on sys.path to import utils
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


# Pydantic Schemas
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
    model_version: str


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
        return {
            "condition": "diabetes",
            "prediction": pred,
            "probability": round(prob, 4),
            "risk_level": "High Risk" if pred == 1 else "Low Risk",
            "model_version": "diabetes-v3.0",
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
        return {
            "condition": "heart",
            "prediction": pred,
            "probability": round(prob, 4),
            "risk_level": "High Risk" if pred == 1 else "Low Risk",
            "model_version": "heart-v3.0",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
