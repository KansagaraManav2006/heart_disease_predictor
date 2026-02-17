"""
Test PDF report generation functionality
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app1 import build_pdf_report


def test_pdf_report_generation():
    """Test that PDF report can be generated successfully"""
    print("Testing PDF report generation...")
    
    # Test data
    disease_name = "Diabetes"
    patient_name = "Test Patient"
    inputs = {
        "Age": 45,
        "Gender": "Male",
        "BMI": 28.5,
        "Smoking History": "never",
        "Hypertension": "No",
        "Heart Disease": "No",
        "HbA1c Level": 6.2,
        "Blood Glucose Level": 140,
    }
    prediction_label = "High Risk"
    probability_percent = 75.5
    
    # Generate PDF
    pdf_bytes = build_pdf_report(
        disease_name=disease_name,
        patient_name=patient_name,
        inputs=inputs,
        prediction_label=prediction_label,
        probability_percent=probability_percent,
    )
    
    # Verify PDF was generated
    assert pdf_bytes is not None, "PDF generation failed - returned None"
    assert isinstance(pdf_bytes, bytes), f"Expected bytes, got {type(pdf_bytes)}"
    assert len(pdf_bytes) > 0, "PDF is empty"
    
    # Check for PDF signature (%PDF-)
    assert pdf_bytes.startswith(b'%PDF-'), "Generated file is not a valid PDF"
    
    print(f"✅ PDF generated successfully ({len(pdf_bytes)} bytes)")


def test_pdf_report_with_empty_name():
    """Test PDF report generation with empty patient name"""
    print("Testing PDF report with empty patient name...")
    
    pdf_bytes = build_pdf_report(
        disease_name="Heart Disease",
        patient_name="",
        inputs={"Age": 50},
        prediction_label="Low Risk",
        probability_percent=25.0,
    )
    
    assert pdf_bytes is not None, "PDF generation failed with empty name"
    assert isinstance(pdf_bytes, bytes), "Expected bytes output"
    print("✅ PDF generated with empty name handled correctly")


def test_pdf_report_heart_disease():
    """Test PDF report for heart disease prediction"""
    print("Testing PDF report for heart disease...")
    
    inputs = {
        "Age": 55,
        "Gender": "Female",
        "Height (cm)": 165,
        "Weight (kg)": 70,
        "BMI": 25.7,
        "Systolic BP (mmHg)": 130,
        "Diastolic BP (mmHg)": 85,
        "Cholesterol (mg/dL)": 220,
        "Glucose (mg/dL)": 110,
        "Smoker": "No",
        "Alcohol Use": "No",
        "Physically Active": "Yes",
    }
    
    pdf_bytes = build_pdf_report(
        disease_name="Heart Disease",
        patient_name="Jane Doe",
        inputs=inputs,
        prediction_label="Low Risk",
        probability_percent=32.8,
    )
    
    assert pdf_bytes is not None, "Heart disease PDF generation failed"
    assert len(pdf_bytes) > 1000, "PDF seems too small"
    print(f"✅ Heart disease PDF generated successfully ({len(pdf_bytes)} bytes)")


if __name__ == "__main__":
    print("=" * 60)
    print("Running PDF Report Generation Tests")
    print("=" * 60)
    print()
    
    try:
        test_pdf_report_generation()
        test_pdf_report_with_empty_name()
        test_pdf_report_heart_disease()
        
        print()
        print("=" * 60)
        print("✅ ALL PDF TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ UNEXPECTED ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
