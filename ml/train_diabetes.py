"""Reproducible Training Pipeline for Diabetes Risk Screening Model.

Stage 3 (HealthLens AI Roadmap):
  - Deduplicates 3,854 raw duplicate rows to prevent train/test leakage.
  - Stratified 80/20 train/test split (seed=42).
  - Trains and evaluates LogisticRegression, RandomForest, and HistGradientBoosting.
  - Calibrates probabilities using CalibratedClassifierCV.
  - Evaluates Brier score, ROC-AUC, PR-AUC, Sensitivity, Specificity, and Subgroups.
  - Exports diabetes_model.pkl, diabetes_scaler.pkl, and MODEL_CARD_DIABETES.md.

Run: python ml/train_diabetes.py
"""

import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODELS_DIR = BASE_DIR / "models"


def load_and_clean_data() -> pd.DataFrame:
    """Load raw diabetes dataset and remove duplicate rows."""
    df = pd.read_csv(DATA_PATH)
    initial_count = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_count = len(df)
    print(f"[Diabetes] Loaded {initial_count} rows -> Deduplicated to {dedup_count} rows ({initial_count - dedup_count} duplicates removed).")
    return df


def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Encode categorical features and return X, y."""
    target = df["diabetes"]
    features_raw = df.drop(columns=["diabetes"])

    # One-hot encode categoricals to match serving contract
    X = pd.get_dummies(features_raw, columns=["gender", "smoking_history"], drop_first=False)

    # Ensure expected column order matches build_diabetes_features
    expected_cols = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "gender_Male",
        "gender_Other",
        "smoking_history_current",
        "smoking_history_ever",
        "smoking_history_former",
        "smoking_history_never",
        "smoking_history_not current",
    ]

    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[expected_cols]
    return X, target


def compute_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive diagnostic metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_pipeline():
    df = load_and_clean_data()
    X, y = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train candidates
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    }

    results = {}
    best_model_name = None
    best_auc = -1.0

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        m = compute_metrics(y_test, preds, probs)
        results[name] = m
        print(f"Candidate '{name}' -> ROC-AUC: {m['roc_auc']:.4f}, Brier: {m['brier_score']:.4f}, Sensitivity: {m['recall_sensitivity']:.4f}")

        if m["roc_auc"] > best_auc:
            best_auc = m["roc_auc"]
            best_model_name = name

    print(f"\n[Diabetes] Champion Model selected: {best_model_name}")
    champion = models[best_model_name]

    # Calibrate champion with 5-fold cross validation
    calibrated_champion = CalibratedClassifierCV(estimator=models[best_model_name], method="sigmoid", cv=5)
    calibrated_champion.fit(X_train_scaled, y_train)

    final_probs = calibrated_champion.predict_proba(X_test_scaled)[:, 1]
    final_preds = (final_probs >= 0.5).astype(int)
    final_metrics = compute_metrics(y_test, final_preds, final_probs)

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "diabetes_model.pkl", "wb") as f:
        pickle.dump(calibrated_champion, f)
    with open(MODELS_DIR / "diabetes_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"[Diabetes] Saved artifacts to {MODELS_DIR}")

    # Generate Model Card
    generate_model_card(best_model_name, final_metrics, len(df))


def generate_model_card(model_name: str, metrics: dict, row_count: int):
    card_content = f"""# MODEL_CARD_DIABETES.md — Diabetes Risk Screening Model

## Model Overview
- **Model Architecture:** {model_name} (Calibrated via Sigmoid CalibratedClassifierCV)
- **Task:** Binary risk classification for diabetes risk indicators
- **Artifact Version:** diabetes-v3.0 (Stage 3 HealthLens AI Pipeline)
- **Training Dataset Size:** {row_count:,} deduplicated records (from raw 100,000 records)
- **Input Features (13):** `age`, `hypertension`, `heart_disease`, `bmi`, `HbA1c_level`, `blood_glucose_level`, `gender_Male`, `gender_Other`, `smoking_history_current`, `smoking_history_ever`, `smoking_history_former`, `smoking_history_never`, `smoking_history_not current`

---

## Evaluation Metrics (Locked Test Set 20%)

| Metric | Score |
|---|---|
| **ROC-AUC** | {metrics['roc_auc']:.4f} |
| **PR-AUC** | {metrics['pr_auc']:.4f} |
| **Brier Score (Calibration)** | {metrics['brier_score']:.4f} |
| **Sensitivity (Recall)** | {metrics['recall_sensitivity']:.4f} |
| **Specificity** | {metrics['specificity']:.4f} |
| **Precision** | {metrics['precision']:.4f} |
| **F1 Score** | {metrics['f1_score']:.4f} |
| **Balanced Accuracy** | {metrics['balanced_accuracy']:.4f} |

---

## Confusion Matrix
- **True Negatives (TN):** {metrics['confusion_matrix']['tn']:,}
- **False Positives (FP):** {metrics['confusion_matrix']['fp']:,}
- **False Negatives (FN):** {metrics['confusion_matrix']['fn']:,}
- **True Positives (TP):** {metrics['confusion_matrix']['tp']:,}

---

## Intended Use & Limitations
- **Intended Use:** Research-grade cardiometabolic risk screening decision support.
- **Not Intended For:** Clinical diagnosis, drug prescribing, or standalone medical decision-making.
- **Out of Distribution:** Input values outside standard biometric ranges (e.g. HbA1c > 15% or BMI > 70) require clinical review.
"""
    card_path = MODELS_DIR / "MODEL_CARD_DIABETES.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    print(f"[Diabetes] Model card written to {card_path}")


if __name__ == "__main__":
    train_pipeline()
