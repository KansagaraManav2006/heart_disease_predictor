# Archived Research Notebooks

> [!NOTE]
> The Jupyter notebooks in this directory represent initial exploratory data analysis and baseline model training.
> They contain legacy machine-specific file paths and uncalibrated models.

In **HealthLens AI Stage 3**, these notebooks are superseded by reproducible Python training pipelines (`ml/train_diabetes.py` and `ml/train_heart.py`) featuring:
- Deterministic random seeding
- Leakage-safe preprocessing & cross-validation
- Leakage cleanup (diabetes row deduplication, heart `id` column removal)
- Model calibration (Brier score & calibration curves)
- Subgroup / fairness metrics
- MLflow experiment tracking & model registration
