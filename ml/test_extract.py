"""Basic tests for the extract.py OCR pipeline.

Run with:  python -m pytest ml/test_extract.py -v
"""
import json
import os
import sys
import tempfile
import unittest

# Ensure the ml directory is importable regardless of where pytest is invoked.
ML_DIR = os.path.dirname(os.path.abspath(__file__))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from extract import parse_medical_data, detect_file_type, validate_file, compute_confidence


class TestParseGlucose(unittest.TestCase):
    def test_glucose_colon(self):
        data, _ = parse_medical_data("Glucose: 150\nHbA1c: 7.2")
        self.assertAlmostEqual(data['glucose'], 150.0)

    def test_glucose_fbs_label(self):
        data, _ = parse_medical_data("FBS: 98\n")
        self.assertAlmostEqual(data['glucose'], 98.0)

    def test_hba1c_extracted(self):
        data, _ = parse_medical_data("HbA1c: 6.5\n")
        self.assertAlmostEqual(data['hba1c'], 6.5)


class TestParseBloodPressure(unittest.TestCase):
    def test_bp_slash_notation(self):
        data, _ = parse_medical_data("BP: 130/85\n")
        self.assertEqual(data['systolic_bp'], 130.0)
        self.assertEqual(data['diastolic_bp'], 85.0)

    def test_blood_pressure_label(self):
        data, _ = parse_medical_data("Blood Pressure: 145/95\n")
        self.assertEqual(data['systolic_bp'], 145.0)


class TestParsePatientName(unittest.TestCase):
    def test_name_on_own_line(self):
        text = "Patient Name: John Smith\nAge: 45\n"
        data, warnings = parse_medical_data(text)
        self.assertIn('patientName', data)
        self.assertEqual(data['patientName'], 'John Smith')

    def test_name_missing_reports_warning(self):
        data, warnings = parse_medical_data("Glucose: 100\n")
        self.assertNotIn('patientName', data)
        self.assertTrue(any('name' in w.lower() for w in warnings))


class TestMissingFields(unittest.TestCase):
    def test_empty_text_all_missing(self):
        data, warnings = parse_medical_data("")
        self.assertEqual(data, {})
        self.assertGreater(len(warnings), 0)

    def test_partial_extraction(self):
        data, _ = parse_medical_data("BMI: 28.5\nAge: 62\n")
        self.assertIn('bmi', data)
        self.assertIn('age', data)
        self.assertNotIn('glucose', data)


class TestDetectFileType(unittest.TestCase):
    def _make_temp_file(self, header_bytes: bytes) -> str:
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(header_bytes)
        f.close()
        return f.name

    def tearDown(self):
        # Clean up any temp files created in tests
        pass

    def test_detect_pdf(self):
        path = self._make_temp_file(b'%PDF-1.4 rest-of-header')
        try:
            self.assertEqual(detect_file_type(path), 'pdf')
        finally:
            os.unlink(path)

    def test_detect_jpeg(self):
        path = self._make_temp_file(b'\xff\xd8\xff\xe0 rest')
        try:
            self.assertEqual(detect_file_type(path), 'jpeg')
        finally:
            os.unlink(path)

    def test_detect_png(self):
        path = self._make_temp_file(b'\x89PNG\r\n\x1a\n rest')
        try:
            self.assertEqual(detect_file_type(path), 'png')
        finally:
            os.unlink(path)

    def test_unknown_type(self):
        path = self._make_temp_file(b'\x00\x00\x00\x00')
        try:
            self.assertEqual(detect_file_type(path), 'unknown')
        finally:
            os.unlink(path)


class TestValidateFile(unittest.TestCase):
    def test_rejects_oversized_file(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        # Write just over 10 MB of valid-looking content
        f.write(b'%PDF')
        f.write(b'x' * (10 * 1024 * 1024 + 1))
        f.close()
        try:
            error = validate_file(f.name)
            self.assertIsNotNone(error)
            self.assertIn('large', error.lower())
        finally:
            os.unlink(f.name)

    def test_rejects_unknown_type(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(b'\x00\x00\x00\x00')
        f.close()
        try:
            error = validate_file(f.name)
            self.assertIsNotNone(error)
            self.assertIn('Unsupported', error)
        finally:
            os.unlink(f.name)

    def test_accepts_valid_pdf_header(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(b'%PDF-1.4\nsome content')
        f.close()
        try:
            error = validate_file(f.name)
            self.assertIsNone(error)
        finally:
            os.unlink(f.name)


class TestComputeConfidence(unittest.TestCase):
    def test_no_fields_is_none(self):
        self.assertEqual(compute_confidence({}, []), 'none')

    def test_few_fields_low(self):
        self.assertEqual(compute_confidence({'glucose': 100}, ['warning']), 'low')

    def test_many_fields_high(self):
        data = {
            'glucose': 100, 'hba1c': 5.5, 'bmi': 24,
            'age': 45, 'cholesterol': 180, 'systolic_bp': 120
        }
        self.assertEqual(compute_confidence(data, []), 'high')


if __name__ == '__main__':
    unittest.main()
