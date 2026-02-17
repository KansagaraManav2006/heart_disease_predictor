"""Smoke tests for module imports and prediction helpers."""

import types

import app
import utils


def test_utils_artifact_loading():
    print("Testing artifact loading...")
    diabetes_model = utils.load_diabetes_model()
    heart_model = utils.load_heart_model()
    diabetes_scaler = utils.load_diabetes_scaler()
    heart_scaler = utils.load_heart_scaler()

    assert hasattr(diabetes_model, "predict"), "Diabetes model missing predict method"
    assert hasattr(diabetes_model, "predict_proba"), "Diabetes model missing predict_proba method"
    assert hasattr(heart_model, "predict"), "Heart model missing predict method"
    assert hasattr(heart_model, "predict_proba"), "Heart model missing predict_proba method"
    assert hasattr(diabetes_scaler, "transform"), "Diabetes scaler missing transform method"
    assert hasattr(heart_scaler, "transform"), "Heart scaler missing transform method"
    print("✅ Artifact loading test passed")


def test_prediction_helpers_return_types():
    print("Testing prediction helper return types...")
    diabetes_model = utils.load_diabetes_model()
    heart_model = utils.load_heart_model()
    diabetes_scaler = utils.load_diabetes_scaler()
    heart_scaler = utils.load_heart_scaler()

    diabetes_features = utils.build_diabetes_features(
        age=30,
        hypertension_opt="No",
        heart_disease_opt="No",
        bmi=25.0,
        hba1c=5.5,
        glucose=100.0,
        gender_opt="Female",
        smoking_opt="never",
    )
    heart_features, bmi_val = utils.build_heart_features(
        age=45,
        gender="Male",
        height_cm=170.0,
        weight_kg=70.0,
        systolic_bp=120,
        diastolic_bp=80,
        cholesterol=200,
        glucose=100,
        smoke=False,
        alco=False,
        active=True,
    )

    diabetes_pred, diabetes_prob = utils.predict_diabetes(
        diabetes_model,
        diabetes_scaler,
        diabetes_features,
    )
    heart_pred, heart_prob = utils.predict_heart(
        heart_model,
        heart_scaler,
        heart_features,
    )

    assert isinstance(diabetes_pred, (int, bool)), f"Diabetes pred should be int/bool, got {type(diabetes_pred)}"
    assert isinstance(diabetes_prob, float), f"Diabetes prob should be float, got {type(diabetes_prob)}"
    assert isinstance(heart_pred, (int, bool)), f"Heart pred should be int/bool, got {type(heart_pred)}"
    assert isinstance(heart_prob, float), f"Heart prob should be float, got {type(heart_prob)}"
    assert isinstance(bmi_val, float), f"BMI should be float, got {type(bmi_val)}"
    print("✅ Prediction helper return types test passed")


def test_app_imports_expose_main():
    print("Testing app main function...")
    assert isinstance(getattr(app, "main"), types.FunctionType), "main should be a function"
    assert callable(app.main), "main should be callable"
    print("✅ App main function test passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Import and Prediction Tests")
    print("="*60 + "\n")
    
    try:
        test_utils_artifact_loading()
        test_prediction_helpers_return_types()
        test_app_imports_expose_main()
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        exit(1)
