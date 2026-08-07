# HealthLens AI — Technical Documentation

This document provides a comprehensive technical breakdown of **HealthLens AI** — an explainable cardiometabolic risk-intelligence and decision-support research platform. It covers data flow, architecture, machine learning contracts, security measures, and research-use disclaimers.

---

> [!IMPORTANT]
> **Research & Decision-Support Prototype**  
> HealthLens AI is built as a research pilot platform for educational, research, and technical portfolio demonstration purposes. It is **not** a FDA-cleared medical device and must **not** be used for clinical diagnosis or treatment.

---

## 1. System Execution & Data Flow

HealthLens AI enforces a secure, validated data pipeline from client submission to risk calculation:

```
[React 19 Client]
       │ (JSON Payload over HTTPS)
       ▼
[Express Gateway / API] ──► [Input Schema Validation & Sanitization]
       │
       ├─► (Session-scoped In-Memory History / PostgreSQL Storage)
       │
       ▼ (Subprocess Timeout & Boundary Guard)
[Python Inference & OCR Engine]
       │
       ├─► PyMuPDF / PyTesseract OCR Extraction
       ├─► Magic-Byte File Type Verification
       ├─► StandardScaler Feature Normalization
       └─► Scikit-Learn Model Inference & Calibrated Probability
```

### Step 1: Input Capture & OCR Review Gate (Frontend)
- **Manual Entry:** Form inputs validate values against health baselines (e.g., positive `height_cm`, blood pressure ranges).
- **OCR Upload:** Lab reports (PDF, JPEG, PNG up to 10 MB) undergo MIME & magic-byte validation. Extracted values are populated into the manual form state and flagged with a **Review Banner**. Prediction is **never auto-triggered**; user confirmation is required.
- **Guided Chat Assistant:** Step-by-step interview interface. Upon answering all questions, input fields and submit actions lock permanently (`isComplete`), prompting form review.

### Step 2: API Gateway & Input Validation (Express Backend)
- Incoming payloads undergo structural validation (e.g. `validateHeartPayload` rejects zero or missing `height_cm` with HTTP 400).
- Uploaded files are capped at 10 MB and restricted to whitelisted MIME types (`application/pdf`, `image/jpeg`, `image/png`).
- Origin-restricted CORS checks prevent unauthorized cross-domain API access.

### Step 3: Inference & Extraction Execution (Python Layer)
- `ml/predict.py` loads scikit-learn models (`diabetes_model.pkl`, `heart_model.pkl`) and scalers.
- `build_diabetes_features` (13 features) and `build_heart_features` (15 features, including `id=0` placeholder until Stage 3 retraining) format inputs to match saved artifact shape.
- `predict_proba()` calculates calibrated risk probability estimates.
- Subprocesses are wrapped with a 30-second timeout and guaranteed temporary file cleanup.

### Step 4: Decision Support & PDF Report Export
- UI displays risk classification (Low vs High) with probability metrics and lifestyle suggestions (`suggestionEngine.js`).
- PDF summaries generated via `jsPDF` include a stable session-based report ID and prominent **"Research Use Only — Not a Clinical Diagnosis"** disclaimers.

---

## 2. Technology Stack & Rationale

| Subsystem | Technology | Rationale |
|---|---|---|
| **Frontend** | React 19.2 + Vite 7 | Sub-second HMR, modern JSX rendering, typed component trees |
| **Styling** | Tailwind CSS v4 | CSS variable design tokens, responsive layout |
| **Backend API** | Node.js 24 + Express 4 | Fast REST gateway, non-blocking I/O, middleware security |
| **ML Engine** | Python 3.10+ & scikit-learn | Proven tabular machine learning pipelines |
| **OCR** | PyMuPDF & PyTesseract | Multipage PDF text extraction and OCR image parsing |
| **Report Export** | jsPDF + autoTable | Client-side vector PDF generation with disclaimers |

---

## 3. Security & Safety Hardening Baseline (Stage 0)

1. **Patient Data Scrubbing:** All mock patient records with personal names removed from repository source.
2. **Subprocess Isolation & Timeout:** 30s execution cap prevents hung processes from exhausting server threads.
3. **Upload File Limits:** 10 MB size limit, magic-byte type checking, and guaranteed temp file deletion.
4. **CORS Restrictions:** Origin whitelist enforces allowed origin domains (`localhost:5173`, `localhost:5000`).
5. **No Raw Text Persistence:** Raw OCR text omitted from API responses to protect sensitive document content.

---

## 4. Platform Roadmap & Evolution

- **Stage 0 (Current):** Repaired baseline, ESLint clean, OCR review gate, payload validation, research disclaimers.
- **Stage 1 (Next):** Express TypeScript refactor, PostgreSQL 17 + Prisma ORM, Argon2id auth, server-side sessions, CSRF protection, RBAC.
- **Stage 2:** Schema-driven assessment forms, patient profiles, clinician access grants.
- **Stage 3:** Leakage-free model retraining (deduplicated diabetes data, heart `id` column removal), internal FastAPI ML service, MLflow registry.
- **Stage 4:** SHAP explainability charts, calibrated risk stratification, versioned reports.
- **Stage 5–10:** Model comparison analytics, pgvector evidence RAG, security hardening, Docker Compose & CI/CD.
