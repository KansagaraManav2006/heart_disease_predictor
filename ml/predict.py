import sys
import os
import json
import traceback

# Set threading and OpenBLAS constraints BEFORE importing numpy/pandas
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_CORETYPE'] = 'HASWELL'

# Ensure the current directory is in the path to find utils.py
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

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

def handle_diabetes(data):
    model = load_diabetes_model()
    scaler = load_diabetes_scaler()
    features = build_diabetes_features(
        age=data.get('age', 0),
        hypertension_opt='Yes' if data.get('hypertension') == '1' else 'No',
        heart_disease_opt='Yes' if data.get('heartDisease') == '1' else 'No',
        bmi=data.get('bmi', 0),
        hba1c=data.get('hba1c', 0),
        glucose=data.get('glucose', 0),
        gender_opt=data.get('gender', '').capitalize(),
        smoking_opt=data.get('smokingHistory', '')
    )
    prediction, probability = predict_diabetes(model, scaler, features)
    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": "High Risk" if prediction == 1 else "Low Risk"
    }

def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def handle_heart(data):
    model = load_heart_model()
    scaler = load_heart_scaler()
    features, bmi_val = build_heart_features(
        age=data.get('age', 0),
        gender=data.get('gender', '').capitalize(),
        # Accept both current frontend keys and legacy payload keys.
        height_cm=data.get('height_cm', data.get('height', 0)),
        weight_kg=data.get('weight_kg', data.get('weight', 0)),
        systolic_bp=data.get('systolic_bp', data.get('systolic', 0)),
        diastolic_bp=data.get('diastolic_bp', data.get('diastolic', 0)),
        cholesterol=data.get('cholesterol', 0),
        glucose=data.get('glucose', data.get('glucose_level', 0)),
        smoke=_to_bool(data.get('smoke')),
        alco=_to_bool(data.get('alco')),
        active=_to_bool(data.get('active'))
    )
    prediction, probability = predict_heart(model, scaler, features)
    return {
        "prediction": prediction,
        "probability": probability,
        "bmi_val": round(bmi_val, 1),
        "risk_level": "High Risk" if prediction == 1 else "Low Risk"
    }

def main():
    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: python predict.py <diabetes|heart> <json_payload>")
        
        disease_type = sys.argv[1]
        payload = json.loads(sys.argv[2])
        
        if disease_type == 'diabetes':
            result = handle_diabetes(payload)
        elif disease_type == 'heart':
            result = handle_heart(payload)
        else:
            raise ValueError(f"Unknown disease type: {disease_type}")
            
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
        sys.exit(1)

if __name__ == "__main__":
    main()
