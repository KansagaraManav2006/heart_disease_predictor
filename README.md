# Disease Prediction System (Diabetes & Heart)

## Setup

1. Create/activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt  # or manually: streamlit pandas numpy scikit-learn matplotlib
   ```

3. Place data under `data/` and models under `models/` (pickles are referenced by absolute paths in `app1.py`).

## Running the app

```bash
streamlit run app1.py
```

The app loads `diabetes_model.pkl`, `heart_model.pkl`, and their scalers, then serves diabetes and heart risk prediction forms plus model metrics.

## Data preparation & training

- Notebooks under `preparation/` clean and prepare datasets (`cleaned_diabetes.csv`, `cleaned_heart.csv`).
- Heart pipeline uses one-hot encoding for `gender`, `cholesterol`, and `gluc` (drop-first). Feature order used for training/prediction: `id, age, height, weight, systolic_bp, diastolic_bp, smoke, alco, active, bmi, gender_2, cholesterol_2, cholesterol_3, gluc_2, gluc_3`.
- Diabetes features used for training/prediction: `age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, gender_Male, gender_Other, smoking_history_current, smoking_history_ever, smoking_history_former, smoking_history_never, smoking_history_not current`.

### Retrain steps (outline)

1. Run the cleaning/prep notebooks to regenerate cleaned data and encoded features.
2. Split into train/validation; fit scaler on X, then fit the classifier.
3. Persist artifacts to `models/` as `*_scaler.pkl` and `*_model.pkl`.
4. Ensure scaler/model expect the exact feature order above before replacing the pickles used by `app1.py`.

## Validating artifacts

Use the helper script to confirm scaler/model alignment with the cleaned data:

```bash
python scripts/validate_models.py
```

This checks that scaler and model input dimensions match the prepared feature matrices for diabetes and heart.

## Notes

- The app shows accuracy, F1, and a confusion matrix computed against the cleaned datasets (cached loads). Large datasets may take a moment on first load.
- If you add another disease (e.g., liver), follow the same pattern: prep notebook → feature list constant → scaler/model → UI branch.

## Usage

1. Create and activate a virtual environment.
2. Install dependencies (see Setup above).
3. Start the app:

   ```bash
   streamlit run app1.py
   ```
