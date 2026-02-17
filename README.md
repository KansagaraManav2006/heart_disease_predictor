# Disease Prediction System (Diabetes & Heart)

A machine learning-powered web application for predicting diabetes and heart disease risk with professional PDF report generation.

## Features

- 🩺 **Dual Disease Prediction**: Diabetes and Heart Disease risk assessment
- 📊 **Interactive UI**: Modern Orange & Gold themed Streamlit interface
- 📄 **PDF Reports**: Generate professional medical reports with patient data
- 🎯 **Accurate Models**: Trained scikit-learn models with scalers
- 🧪 **Comprehensive Testing**: Unit tests for all major components
- 📦 **Modular Architecture**: Separated business logic (`utils.py`) from UI (`app.py`)

## Setup

1. Create/activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Core dependencies:
   - `streamlit>=1.31.0` - Web UI framework
   - `pandas>=2.0.0` - Data manipulation
   - `numpy>=1.24.0` - Numerical computing
   - `scikit-learn>=1.3.0` - Machine learning models
   - `matplotlib>=3.7.0` - Visualization
   - `fpdf>=1.7.2` - PDF report generation

3. Place data under `data/` and models under `models/` (pickles are referenced by absolute paths in `app.py`).

## Running the app

```bash
streamlit run app.py
```

The app loads `diabetes_model.pkl`, `heart_model.pkl`, and their scalers, then serves diabetes and heart risk prediction forms with:

- Patient name input for personalized reports
- Real-time risk prediction with probability scores
- Professional PDF report generation
- Interactive visualizations and progress bars

## Testing

Run the test suite to verify all functionality:

```bash
# Test imports and prediction pipeline
python -m tests.test_imports

# Test PDF report generation
python -m tests.test_pdf_report
```

Test coverage includes:

- ✅ Artifact loading (models and scalers)
- ✅ Prediction function return types
- ✅ Feature engineering for both diseases
- ✅ PDF report generation with various inputs
- ✅ Main application entry point

## Project Structure

```text
.
├── app.py                   # Main Streamlit application (UI layer)
├── utils.py                 # Business logic (model loading, predictions, feature engineering)
├── requirements.txt         # Python dependencies
├── models/                  # Trained ML models and scalers
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   └── heart_scaler.pkl
├── data/                    # Training and cleaned datasets
│   ├── cleaned_diabetes.csv
│   ├── cleaned_heart.csv
│   ├── diabetes.csv
│   └── heart.csv
├── tests/                   # Test suite
│   ├── test_imports.py     # Import and prediction tests
│   └── test_pdf_report.py  # PDF generation tests
├── preparation/             # Data preparation notebooks
│   ├── clean_diab.ipynb
│   ├── clean_heart.ipynb
│   ├── diabetes_data_preparation.ipynb
│   └── heart_data_preparation.ipynb
└── models/                  # Model training notebooks
    ├── diabetes_model.ipynb
    └── heart_model.ipynb
```

## Data preparation & training

- Notebooks under `preparation/` clean and prepare datasets (`cleaned_diabetes.csv`, `cleaned_heart.csv`).
- Heart pipeline uses one-hot encoding for `gender`, `cholesterol`, and `gluc` (drop-first). Feature order used for training/prediction: `id, age, height, weight, systolic_bp, diastolic_bp, smoke, alco, active, bmi, gender_2, cholesterol_2, cholesterol_3, gluc_2, gluc_3`.
- Diabetes features used for training/prediction: `age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, gender_Male, gender_Other, smoking_history_current, smoking_history_ever, smoking_history_former, smoking_history_never, smoking_history_not current`.

### Retrain steps (outline)

1. Run the cleaning/prep notebooks to regenerate cleaned data and encoded features.
2. Split into train/validation; fit scaler on X, then fit the classifier.
3. Persist artifacts to `models/` as `*_scaler.pkl` and `*_model.pkl`.
4. Ensure scaler/model expect the exact feature order above before replacing the pickles used by `app.py`.

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
   streamlit run app.py
   ```

4. In the web interface:
   - Enter patient name (optional, but recommended for reports)
   - Select disease type (Diabetes or Heart Disease)
   - Fill in all required health parameters
   - Click "INITIATE SCAN" to get prediction
   - Download PDF report with detailed analysis

## PDF Report Features

The generated PDF reports include:

- **Header**: Disease Prediction Report title with professional formatting
- **Patient Information**: Name, generation timestamp, and condition type
- **Input Parameters**: Complete list of all submitted health metrics
- **Results**: Risk prediction (High/Low Risk) with probability percentage
- **Recommendations**: Basic guidance based on risk level

Reports are automatically named using the patient's name for easy organization.

## Architecture

The application follows a clean, modular architecture:

### app.py - UI Layer

- Streamlit-based user interface
- Custom Orange & Gold theme with animations
- Modular render functions for each section
- Patient name input and disease selection
- PDF download buttons

Key functions:
- `load_artifacts()` - Load all models and scalers with error handling
- `inject_theme()` - Apply custom CSS styling
- `render_header()` - Display main header
- `render_sidebar()` - Show sidebar information
- `render_diabetes_section()` - Diabetes prediction UI
- `render_heart_section()` - Heart disease prediction UI
- `render_footer()` - Application footer
- `build_pdf_report()` - Generate PDF reports
- `main()` - Application entry point

### utils.py - Business Logic Layer

- Model and scaler loading with error handling
- Feature engineering for both diseases
- Prediction wrapper functions

Key functions:
- `load_diabetes_model()` / `load_heart_model()` - Load trained models
- `load_diabetes_scaler()` / `load_heart_scaler()` - Load fitted scalers
- `build_diabetes_features()` - Convert UI inputs to DataFrame for diabetes
- `build_heart_features()` - Convert UI inputs with one-hot encoding for heart disease
- `predict_diabetes()` / `predict_heart()` - Make predictions and return probabilities

Custom exception:
- `ArtifactLoadError` - Raised when model/scaler loading fails

## Development

### Code Style

- Follow PEP 8 guidelines for Python code
- Use type hints where applicable
- Keep functions focused and single-purpose
- Separate UI logic from business logic

### Adding New Predictions

To add a new disease prediction:
1. Create data preparation notebook in `preparation/`
2. Train model and save scaler + model pickles in `models/`
3. Add loader functions in `utils.py`
4. Add feature builder function with proper encoding
5. Add prediction wrapper function
6. Create render section in `app.py`
7. Add tests in `tests/`

### Contributing

When making changes:
1. Run tests: `python -m tests.test_imports` and `python -m tests.test_pdf_report`
2. Verify no syntax errors: `python -m py_compile app.py utils.py`
3. Test the app: `streamlit run app.py`
4. Update documentation as needed

## License

Educational project for machine learning demonstration.

## Credits

Built by Smit Kansagara
