"""Utility helpers for the disease prediction Streamlit app."""

from pathlib import Path
from typing import Any, Tuple
import pickle

import pandas as pd


class ArtifactLoadError(RuntimeError):
    """Raised when a persisted model or scaler artifact cannot be loaded."""


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def _load_artifact(filename: str) -> Any:
    """Load a pickle artifact from the models directory with error handling."""
    artifact_path = MODELS_DIR / filename
    try:
        with open(artifact_path, "rb") as artifact_file:
            return pickle.load(artifact_file)
    except FileNotFoundError as exc:
        raise ArtifactLoadError(f"Missing artifact: {artifact_path}") from exc
    except pickle.UnpicklingError as exc:
        raise ArtifactLoadError(f"Corrupted artifact: {artifact_path}") from exc


def load_diabetes_model() -> Any:
    """Return the trained diabetes prediction model."""
    return _load_artifact("diabetes_model.pkl")


def load_heart_model() -> Any:
    """Return the trained heart disease prediction model."""
    return _load_artifact("heart_model.pkl")


def load_diabetes_scaler() -> Any:
    """Return the persisted scaler for diabetes features."""
    return _load_artifact("diabetes_scaler.pkl")


def load_heart_scaler() -> Any:
    """Return the persisted scaler for heart disease features."""
    return _load_artifact("heart_scaler.pkl")


def build_diabetes_features(
    *,
    age: float,
    hypertension_opt: str,
    heart_disease_opt: str,
    bmi: float,
    hba1c: float,
    glucose: float,
    gender_opt: str,
    smoking_opt: str,
) -> pd.DataFrame:
    """Compose the diabetes feature frame from raw UI inputs."""
    hypertension = 1 if hypertension_opt == "Yes" else 0
    heart_disease = 1 if heart_disease_opt == "Yes" else 0
    gender_male = 1 if gender_opt == "Male" else 0
    gender_other = 1 if gender_opt == "Other" else 0

    smoking_current = 1 if smoking_opt == "current" else 0
    smoking_ever = 1 if smoking_opt == "ever" else 0
    smoking_former = 1 if smoking_opt == "former" else 0
    smoking_never = 1 if smoking_opt == "never" else 0
    smoking_not_current = 1 if smoking_opt == "not current" else 0

    feature_row = {
        "age": float(age),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "bmi": float(bmi),
        "HbA1c_level": float(hba1c),
        "blood_glucose_level": float(glucose),
        "gender_Male": int(gender_male),
        "gender_Other": int(gender_other),
        "smoking_history_current": int(smoking_current),
        "smoking_history_ever": int(smoking_ever),
        "smoking_history_former": int(smoking_former),
        "smoking_history_never": int(smoking_never),
        "smoking_history_not current": int(smoking_not_current),
    }

    return pd.DataFrame([feature_row])


def build_heart_features(
    *,
    age: float,
    gender: str,
    height_cm: float,
    weight_kg: float,
    systolic_bp: float,
    diastolic_bp: float,
    cholesterol: float,
    glucose: float,
    smoke: bool,
    alco: bool,
    active: bool,
) -> Tuple[pd.DataFrame, float]:
    """Compose the heart disease feature frame and BMI from raw UI inputs."""
    gender_val = 1 if gender == "Male" else 2
    bmi_val = float(weight_kg) / ((float(height_cm) / 100.0) ** 2)

    if cholesterol < 200:
        cholesterol_cat = 1
    elif cholesterol < 240:
        cholesterol_cat = 2
    else:
        cholesterol_cat = 3

    if glucose < 100:
        glucose_cat = 1
    elif glucose < 126:
        glucose_cat = 2
    else:
        glucose_cat = 3

    feature_row = {
        "id": 0,
        "age": float(age),
        "gender": int(gender_val),
        "height": int(height_cm),
        "weight": float(weight_kg),
        "systolic_bp": int(systolic_bp),
        "diastolic_bp": int(diastolic_bp),
        "cholesterol": int(cholesterol_cat),
        "gluc": int(glucose_cat),
        "smoke": int(bool(smoke)),
        "alco": int(bool(alco)),
        "active": int(bool(active)),
        "bmi": float(bmi_val),
    }

    df = pd.DataFrame([feature_row])
    
    # One-hot encode categorical features to match training format
    df = pd.get_dummies(df, columns=["gender", "cholesterol", "gluc"], drop_first=True)
    
    # Ensure all expected columns exist (even if zero)
    expected_columns = [
        "id", "age", "height", "weight", "systolic_bp", "diastolic_bp",
        "smoke", "alco", "active", "bmi",
        "gender_2", "cholesterol_2", "cholesterol_3", "gluc_2", "gluc_3"
    ]
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match expected order
    df = df[expected_columns]
    
    return df, bmi_val


def predict_diabetes(model: Any, scaler: Any, features: pd.DataFrame) -> Tuple[int, float]:
    """Return the diabetes prediction label and probability."""
    user_scaled = scaler.transform(features)
    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]
    return int(prediction), float(probability)


def predict_heart(model: Any, scaler: Any, features: pd.DataFrame) -> Tuple[int, float]:
    """Return the heart disease prediction label and probability."""
    user_scaled = scaler.transform(features)
    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]
    return int(prediction), float(probability)
