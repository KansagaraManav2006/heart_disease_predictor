# HealthLens AI — Explainable Cardiometabolic Risk Intelligence Platform

> **Research & Decision-Support Prototype**  
> *For research, educational, and portfolio demonstration purposes only. Not a regulated medical device or clinical diagnostic tool.*

HealthLens AI is an explainable cardiometabolic risk-intelligence and decision-support research platform for evaluating **Diabetes** and **Cardiovascular Disease** risk profiles.

---

## ⚠️ Medical & Regulatory Disclaimer

- **Not a Diagnostic Tool:** HealthLens AI generates research-grade statistical risk estimates based on machine learning models trained on public research datasets. It does **not** provide clinical diagnosis, medical treatment, or prescription advice.
- **FDA / Health Authority Guidance:** Under FDA Clinical Decision Support software guidelines, patient-facing risk prediction tools are subject to regulatory oversight prior to clinical use. Public clinical deployment requires formal validation and regulatory clearance.
- **Consult Healthcare Professionals:** Patients must consult qualified healthcare professionals for medical evaluations.

---

## 🌟 Key Features

- 🩺 **Dual-Domain Cardiometabolic Risk Assessment:** Diabetes (13 parameters) and Cardiovascular (15 parameters) risk scoring using scikit-learn models.
- 🔍 **OCR-Assisted Data Extraction (Review-Gated):** Extract laboratory measurements from PDFs/images via PyMuPDF and Tesseract OCR. Extracted parameters require explicit user verification before running predictions.
- 📊 **Calibrated Risk Stratification:** Provides low, moderate, and high risk classification paired with calibrated probability estimates.
- 📄 **Research Report Generation:** Export summary reports (PDF) featuring stable report IDs and research disclaimers.
- 💬 **Guided Assessment Assistant:** Step-by-step chat flow to complete assessment parameters with input locking upon completion.
- 📱 **Responsive Medical UI:** Accessible interface built with React 19, Tailwind CSS v4, and custom WCAG-compliant design tokens.

---

## 🏗️ System Architecture

```text
HealthLens AI Platform
├── client/                 React 19 + Vite 7 Frontend (Tailwind CSS v4)
├── server/                 Node.js 24 + Express Backend (Auth, Sessions, Validation)
├── ml/                     Python Inference & OCR Utilities (scikit-learn, PyMuPDF, PyTesseract)
│   ├── models/             Trained ML artifacts (.pkl)
│   └── test_fixtures/      Sample PDFs and images for automated testing
├── research/
│   └── archive/notebooks/  Archived Jupyter EDA notebooks (research reference)
└── docs/                   Architecture Decision Records (ADRs) & Specification Docs
```

---

## 💻 Setup & Local Development

### 1. Prerequisites
- **Node.js**: v20+ (LTS recommended)
- **Python**: v3.10+
- **Tesseract OCR** (Optional, for image OCR extraction):
  - Windows: [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (Add to PATH)
  - macOS: `brew install tesseract`
  - Linux: `sudo apt install tesseract-ocr`

### 2. Environment Configuration
Copy the `.env.example` file to create your local `.env`:
```bash
cp .env.example .env
cp server/.env.example server/.env
cp client/.env.example client/.env
```

### 3. Installation
```bash
# Install Node dependencies across root, server, and client
npm run install:all

# Install Python dependencies
pip install -r ml/requirements.txt
```

### 4. Run Development Server
```bash
npm run dev
```
- **Frontend App:** `http://localhost:5173`
- **Express API:** `http://localhost:5000`

---

## 🧪 Testing

```bash
# Run client linter (ESLint)
cd client && npx eslint .

# Build client production bundle
cd client && npm run build

# Run Python unit & OCR tests
python -m unittest discover -s ml -p "test_*.py"
```

---

## 📜 Documentation

- [`PROJECT_AUDIT_REPORT.md`](./PROJECT_AUDIT_REPORT.md) — Comprehensive security audit & verified findings report.
- [`DOCUMENTATION.md`](./DOCUMENTATION.md) — Detailed technical architecture and domain specifications.
- [`docs/ADR-001-architecture.md`](./docs/ADR-001-architecture.md) — Architecture Decision Record.
