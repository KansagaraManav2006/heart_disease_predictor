"""Unit & Contract Tests for Stage 3 ML Pipelines & FastAPI Inference Service.

Run with: python -m unittest discover -s ml -p "test_*.py" -v
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure ml directory is importable
ML_DIR = Path(__file__).resolve().parent
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


class TestStage3MLPipelines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diabetes_model = load_diabetes_model()
        cls.diabetes_scaler = load_diabetes_scaler()
        cls.heart_model = load_heart_model()
        cls.heart_scaler = load_heart_scaler()

    def test_diabetes_feature_vector_shape(self):
        df = build_diabetes_features(
            age=45,
            hypertension_opt="No",
            heart_disease_opt="No",
            bmi=25.0,
            hba1c=5.5,
            glucose=90,
            gender_opt="Male",
            smoking_opt="never",
        )
        self.assertEqual(df.shape, (1, 13))
        self.assertNotIn("id", df.columns)

    def test_heart_feature_vector_shape_no_id(self):
        df, bmi = build_heart_features(
            age=50,
            gender="Male",
            height_cm=170,
            weight_kg=70,
            systolic_bp=120,
            diastolic_bp=80,
            cholesterol=190,
            glucose=95,
            smoke=False,
            alco=False,
            active=True,
        )
        self.assertEqual(df.shape, (1, 14))
        self.assertNotIn("id", df.columns)
        self.assertAlmostEqual(bmi, 24.22, places=2)

    def test_diabetes_prediction(self):
        df = build_diabetes_features(
            age=65,
            hypertension_opt="Yes",
            heart_disease_opt="Yes",
            bmi=32.0,
            hba1c=8.5,
            glucose=200,
            gender_opt="Female",
            smoking_opt="current",
        )
        pred, prob = predict_diabetes(
            self.diabetes_model, self.diabetes_scaler, df
        )
        self.assertIn(pred, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertEqual(pred, 1)

    def test_heart_prediction(self):
        df, _ = build_heart_features(
            age=60,
            gender="Male",
            height_cm=165,
            weight_kg=90,
            systolic_bp=160,
            diastolic_bp=100,
            cholesterol=260,
            glucose=140,
            smoke=True,
            alco=True,
            active=False,
        )
        pred, prob = predict_heart(
            self.heart_model, self.heart_scaler, df
        )
        self.assertIn(pred, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertEqual(pred, 1)

    def test_heart_zero_height_rejected(self):
        with self.assertRaises(ValueError):
            build_heart_features(
                age=40,
                gender="Male",
                height_cm=0,
                weight_kg=70,
                systolic_bp=120,
                diastolic_bp=80,
                cholesterol=180,
                glucose=90,
                smoke=False,
                alco=False,
                active=True,
            )


class TestFastAPIEndpoints(unittest.TestCase):
    def test_fastapi_imports(self):
        from ml.api.main import app
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
