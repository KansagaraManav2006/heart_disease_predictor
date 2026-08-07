# MODEL_CARD_DIABETES.md — Diabetes Risk Screening Model

## Model Overview
- **Model Architecture:** HistGradientBoosting (Calibrated via Sigmoid CalibratedClassifierCV)
- **Task:** Binary risk classification for diabetes risk indicators
- **Artifact Version:** diabetes-v3.0 (Stage 3 HealthLens AI Pipeline)
- **Training Dataset Size:** 96,146 deduplicated records (from raw 100,000 records)
- **Input Features (13):** `age`, `hypertension`, `heart_disease`, `bmi`, `HbA1c_level`, `blood_glucose_level`, `gender_Male`, `gender_Other`, `smoking_history_current`, `smoking_history_ever`, `smoking_history_former`, `smoking_history_never`, `smoking_history_not current`

---

## Evaluation Metrics (Locked Test Set 20%)

| Metric | Score |
|---|---|
| **ROC-AUC** | 0.9769 |
| **PR-AUC** | 0.8805 |
| **Brier Score (Calibration)** | 0.0235 |
| **Sensitivity (Recall)** | 0.6863 |
| **Specificity** | 0.9994 |
| **Precision** | 0.9915 |
| **F1 Score** | 0.8111 |
| **Balanced Accuracy** | 0.8429 |

---

## Confusion Matrix
- **True Negatives (TN):** 17,524
- **False Positives (FP):** 10
- **False Negatives (FN):** 532
- **True Positives (TP):** 1,164

---

## Intended Use & Limitations
- **Intended Use:** Research-grade cardiometabolic risk screening decision support.
- **Not Intended For:** Clinical diagnosis, drug prescribing, or standalone medical decision-making.
- **Out of Distribution:** Input values outside standard biometric ranges (e.g. HbA1c > 15% or BMI > 70) require clinical review.
