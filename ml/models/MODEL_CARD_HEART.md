# MODEL_CARD_HEART.md — Heart Disease Risk Screening Model

## Model Overview
- **Model Architecture:** HistGradientBoosting (Calibrated via Sigmoid CalibratedClassifierCV)
- **Task:** Binary risk classification for cardiovascular disease indicators
- **Artifact Version:** heart-v3.0 (Stage 3 HealthLens AI Pipeline)
- **Training Dataset Size:** 70,000 records (dataset `id` feature removed)
- **Input Features (14):** `age`, `height`, `weight`, `systolic_bp`, `diastolic_bp`, `smoke`, `alco`, `active`, `bmi`, `gender_2`, `cholesterol_2`, `cholesterol_3`, `gluc_2`, `gluc_3`

---

## Evaluation Metrics (Locked Test Set 20%)

| Metric | Score |
|---|---|
| **ROC-AUC** | 0.8001 |
| **PR-AUC** | 0.7836 |
| **Brier Score (Calibration)** | 0.1809 |
| **Sensitivity (Recall)** | 0.6988 |
| **Specificity** | 0.7706 |
| **Precision** | 0.7526 |
| **F1 Score** | 0.7247 |
| **Balanced Accuracy** | 0.7347 |

---

## Confusion Matrix
- **True Negatives (TN):** 5,397
- **False Positives (FP):** 1,607
- **False Negatives (FN):** 2,107
- **True Positives (TP):** 4,889

---

## Intended Use & Limitations
- **Intended Use:** Research-grade cardiometabolic risk screening decision support.
- **Data Leakage Fix:** Dataset row `id` column removed entirely from feature space.
- **Not Intended For:** Clinical diagnosis, drug prescribing, or standalone medical decision-making.
