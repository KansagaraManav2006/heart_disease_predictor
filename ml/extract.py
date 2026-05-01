import sys
import json
import re
import os

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    if ext == '.pdf':
        try:
            import fitz
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except ImportError:
            return {"error": "PyMuPDF (fitz) is not installed. Please run pip install PyMuPDF"}
        except Exception as e:
            return {"error": str(e)}
    elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
        try:
            from PIL import Image
            import pytesseract
            # Note: Tesseract-OCR binary must be installed on the system and in PATH
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        except ImportError:
            return {"error": "Pillow or pytesseract is not installed."}
        except Exception as e:
            return {"error": f"Image processing failed. Is Tesseract installed? Details: {str(e)}"}
    else:
        # Fallback by attempting to parse the file as pdf or image anyway, assuming extension might be missing
        # We will assume it's an image first, since multer often uploads without extensions
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        except:
            try:
                import fitz
                doc = fitz.open(file_path)
                for page in doc:
                    text += page.get_text()
                doc.close()
            except Exception as e:
                return {"error": "Unsupported file type and could not resolve automatically."}
                
    return text

def parse_medical_data(text):
    data = {}
    
    # Text sanitization
    # Replace common OCR misreads
    text = text.replace('\n', ' ')
    
    # 1. Glucose
    glucose_match = re.search(r'(?i)(?:glucose|sugar|fbs)[\s\.\,\:\=]+(\d+(?:\.\d+)?)', text)
    if glucose_match:
        data['glucose'] = float(glucose_match.group(1))
        
    # 2. HbA1c
    hba1c_match = re.search(r'(?i)(?:hba1c|a1c|hemoglobin a1c)[\s\.\,\:\=]+(\d+(?:\.\d+)?)', text)
    if hba1c_match:
        data['hba1c'] = float(hba1c_match.group(1))
        
    # 3. BP (Systolic and Diastolic)
    bp_match = re.search(r'(?i)(?:bp|blood pressure)[\s\.\,\:\=]*(\d{2,3})\s*[/\\\|\-]\s*(\d{2,3})', text)
    if bp_match:
        data['systolic_bp'] = float(bp_match.group(1))
        data['diastolic_bp'] = float(bp_match.group(2))
        
    # 4. BMI
    bmi_match = re.search(r'(?i)(?:bmi|body mass index)[\s\.\,\:\=]+(\d{2}(?:\.\d+)?)', text)
    if bmi_match:
        data['bmi'] = float(bmi_match.group(1))
        
    # 5. Age
    age_match = re.search(r'(?i)age[\s\.\,\:\=]+(\d{1,3})', text)
    if age_match:
        data['age'] = int(age_match.group(1))
        
    # 6. Cholesterol
    chol_match = re.search(r'(?i)(?:cholesterol|chol)[\s\.\,\:\=]+(\d+(?:\.\d+)?)', text)
    if chol_match:
        data['cholesterol'] = float(chol_match.group(1))

    return data

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No file path provided"}))
        sys.exit(1)
        
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(json.dumps({"error": "File not found"}))
        sys.exit(1)
        
    text_result = extract_text(file_path)
    
    if isinstance(text_result, dict) and "error" in text_result:
        print(json.dumps(text_result))
        sys.exit(1)
        
    parsed_data = parse_medical_data(text_result)
    
    # Also return the raw text mostly for debugging or future-proofing
    out = {
        "success": True,
        "extracted_data": parsed_data,
        "raw_text": text_result[:500] + "..." if len(text_result) > 500 else text_result
    }
    
    print(json.dumps(out))
    
if __name__ == "__main__":
    main()
