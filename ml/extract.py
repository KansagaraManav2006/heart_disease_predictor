import sys
import json
import re
import os
import struct

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_CORETYPE'] = 'HASWELL'

# Maximum file size the server should enforce before calling this script.
# This is a defence-in-depth guard at the Python layer.
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_MIME_TYPES = {'pdf', 'jpeg', 'png'}


def detect_file_type(file_path: str) -> str:
    """Detect file type from magic bytes, NOT the filename extension.

    Returns one of: 'pdf', 'jpeg', 'png', or 'unknown'.
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
    except OSError:
        return 'unknown'

    if header[:4] == b'%PDF':
        return 'pdf'
    if header[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    return 'unknown'


def validate_file(file_path: str) -> str | None:
    """Return an error string if the file is invalid, else None."""
    try:
        size = os.path.getsize(file_path)
    except OSError as e:
        return f"Cannot read file: {e}"

    if size > MAX_FILE_SIZE_BYTES:
        mb = size / 1024 / 1024
        return f"File too large ({mb:.1f} MB). Maximum allowed size is 10 MB."

    ftype = detect_file_type(file_path)
    if ftype not in ALLOWED_MIME_TYPES:
        return (
            f"Unsupported file type (detected: {ftype!r}). "
            "Only PDF, JPEG, and PNG are accepted."
        )
    return None


def extract_text(file_path: str) -> tuple[str, str]:
    """Extract raw text from a PDF or image file.

    Returns (text, file_type). Raises RuntimeError on failure.
    """
    ftype = detect_file_type(file_path)

    if ftype == 'pdf':
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError(
                "PyMuPDF is not installed. Run: pip install PyMuPDF"
            )
        try:
            doc = fitz.open(file_path)
            # Use 'text' mode to preserve line breaks (critical for name parsing).
            pages = [page.get_text('text') for page in doc]
            doc.close()
            return '\n'.join(pages), 'pdf'
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed: {e}")

    if ftype in ('jpeg', 'png'):
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            raise RuntimeError(
                "Pillow or pytesseract is not installed. "
                "Run: pip install Pillow pytesseract"
            )
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            return text, ftype
        except Exception as e:
            raise RuntimeError(
                f"Image OCR failed. Is Tesseract installed and in PATH? Details: {e}"
            )

    raise RuntimeError(
        f"Unsupported file type: {ftype!r}. Only PDF, JPEG, and PNG are accepted."
    )


# ---------------------------------------------------------------------------
# Required fields per disease type (for missing-field reporting)
# ---------------------------------------------------------------------------
DIABETES_REQUIRED = ['glucose', 'hba1c']
HEART_REQUIRED = ['systolic_bp', 'diastolic_bp']
ALL_EXTRACTABLE = ['glucose', 'hba1c', 'systolic_bp', 'diastolic_bp',
                   'bmi', 'age', 'cholesterol', 'patientName']


def parse_medical_data(text: str) -> dict:
    """Extract structured fields from OCR text.

    Newline boundaries are PRESERVED before calling this function.
    Only targeted whitespace (leading/trailing per line) is normalised.
    """
    data = {}
    warnings = []

    # Normalise: trim each line but preserve the line structure for regex.
    lines = [line.strip() for line in text.splitlines()]
    # Also create a single-line version for patterns that span whitespace.
    single = ' '.join(line for line in lines if line)

    # --- 1. Glucose ---
    m = re.search(
        r'(?i)(?:glucose|sugar|fbs)\s*[:\=\.\,]?\s*(\d+(?:\.\d+)?)',
        single
    )
    if m:
        data['glucose'] = float(m.group(1))

    # --- 2. HbA1c ---
    m = re.search(
        r'(?i)(?:hba1c|a1c|hemoglobin\s+a1c)\s*[:\=\.\,]?\s*(\d+(?:\.\d+)?)',
        single
    )
    if m:
        data['hba1c'] = float(m.group(1))

    # --- 3. Blood Pressure (Systolic / Diastolic) ---
    m = re.search(
        r'(?i)(?:bp|blood\s*pressure)\s*[:\=\.\,]?\s*(\d{2,3})\s*[/\\\|\-]\s*(\d{2,3})',
        single
    )
    if m:
        data['systolic_bp'] = float(m.group(1))
        data['diastolic_bp'] = float(m.group(2))

    # --- 4. BMI ---
    m = re.search(
        r'(?i)(?:bmi|body\s*mass\s*index)\s*[:\=\.\,]?\s*(\d{2}(?:\.\d+)?)',
        single
    )
    if m:
        data['bmi'] = float(m.group(1))

    # --- 5. Age ---
    m = re.search(
        r'(?i)age\s*[:\=\.\,]?\s*(\d{1,3})',
        single
    )
    if m:
        data['age'] = int(m.group(1))

    # --- 6. Cholesterol ---
    m = re.search(
        r'(?i)(?:cholesterol|chol)\s*[:\=\.\,]?\s*(\d+(?:\.\d+)?)',
        single
    )
    if m:
        data['cholesterol'] = float(m.group(1))

    # --- 7. Patient Name (multiline-aware) ---
    # Try each line: look for "Patient Name:" or "Name:" label followed by text.
    name_pattern = re.compile(
        r'(?i)(?:patient\s+name|name)\s*[:\=\.\,]\s*([A-Za-z][A-Za-z ]{1,38}?)(?:\s*(?:\n|\r|$|\b(?:Age|DOB|Date|Gender|ID|MRN|Phone|Email|Address|\d)))',
        re.MULTILINE
    )
    m = name_pattern.search(text)  # use original text to respect line breaks
    if m:
        candidate = m.group(1).strip()
        # Reject if it looks like a section header word
        if len(candidate.split()) >= 1 and len(candidate) <= 40:
            data['patientName'] = candidate
        else:
            warnings.append("Patient name found but appeared invalid; skipped.")
    else:
        warnings.append("Patient name not detected in document.")

    return data, warnings


def compute_confidence(extracted: dict, warnings: list) -> str:
    """Return a rough confidence tier based on how many fields were extracted."""
    found = len([k for k in ALL_EXTRACTABLE if k in extracted])
    if found == 0:
        return 'none'
    if found <= 2 or warnings:
        return 'low'
    if found <= 4:
        return 'medium'
    return 'high'


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No file path provided"}))
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(json.dumps({"error": "File not found"}))
        sys.exit(1)

    # Validate size and type before doing any heavy processing
    validation_error = validate_file(file_path)
    if validation_error:
        print(json.dumps({"error": validation_error}))
        sys.exit(1)

    try:
        text, detected_type = extract_text(file_path)
    except RuntimeError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    extracted_data, warnings = parse_medical_data(text)

    # Compute which required fields are still missing
    missing_diabetes = [f for f in DIABETES_REQUIRED if f not in extracted_data]
    missing_heart = [f for f in HEART_REQUIRED if f not in extracted_data]
    confidence = compute_confidence(extracted_data, warnings)

    # NOTE: raw_text is intentionally NOT returned — we never persist or
    # transmit raw medical text to the client.
    out = {
        "success": True,
        "detected_type": detected_type,
        "extracted_data": extracted_data,
        "confidence": confidence,
        "warnings": warnings,
        "missing_fields": {
            "diabetes": missing_diabetes,
            "heart": missing_heart,
        },
    }

    print(json.dumps(out))


if __name__ == "__main__":
    main()
