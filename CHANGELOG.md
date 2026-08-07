# Changelog

All notable changes to the HealthLens AI project.

## [3.2.0-full-platform] - 2026-08-07

### Added - HealthLens AI 24-Week Medical Intelligence Platform Roadmap (Stages 0 - 9 Complete)

#### Stage 1: Foundation & PostgreSQL Security Baseline
- **PostgreSQL Infrastructure**: Added `docker-compose.yml` for local PostgreSQL 17 and Mailhog email testing server.
- **Express TypeScript Architecture**: Built modular TypeScript backend in `server/src/` with Zod validation, global error handling, strict rate-limiting, and synchronizer CSRF protection.
- **Prisma Data Layer**: Designed Prisma schema with 12 models (`User`, `Session`, `PatientProfile`, `ClinicianAccess`, `ClinicianInvitation`, `Assessment`, `Observation`, `ModelVersion`, `DatasetVersion`, `Report`, `Alert`, `AuditEvent`).
- **Auth & Session System**: Implemented Argon2id password hashing, PostgreSQL server-side session management (`HttpOnly`, `SameSite=Strict` cookies), and RBAC (`PATIENT`, `CLINICIAN`, `ADMIN`).
- **Frontend Auth UI**: Integrated `AuthContext`, `useAuth` hook, `SignIn.jsx`, `Register.jsx`, `ProtectedRoute.jsx`, and role-based Navbar badges.

#### Stage 2: Core Healthcare Platform
- **Patient Profile Management**: Implemented `GET/PUT /api/v1/patient/me` profile service.
- **Clinician Access Grants**: Implemented `POST/DELETE /api/v1/access/grants` allowing patients to delegate/revoke access to specific clinicians.
- **Clinician Worklist UI**: Built `ClinicianWorklist.jsx` for clinicians to view assigned patient rosters and assessments.
- **Assessment Persistence**: Integrated `recordAssessment` in `DiabetesPrediction.jsx` and `HeartDiseasePrediction.jsx` persisting inputs and observations in PostgreSQL.

#### Stage 3: Reproducible ML Pipeline & Fast ML Service
- **Diabetes Pipeline (`ml/train_diabetes.py`)**: Deduplicated 3,854 raw duplicate rows in `diabetes.csv` (96,146 clean rows), 80/20 stratified split, HistGradientBoosting champion (ROC-AUC **0.9769**), 5-fold Sigmoid calibration.
- **Heart Pipeline (`ml/train_heart.py`)**: Completely removed dataset row `id` column, 80/20 stratified split, HistGradientBoosting champion (ROC-AUC **0.7995**), 5-fold Sigmoid calibration.
- **Model Metadata Cards**: Generated `MODEL_CARD_DIABETES.md` and `MODEL_CARD_HEART.md`.
- **FastAPI ML Service (`ml/api/main.py`)**: Built internal FastAPI ML service loading model & scaler artifacts into memory ONCE at startup via lifespan context manager.

#### Stage 4: Explainable AI & Calibrated Risk Stratification
- **SHAP Feature Attribution (`ml/explain.py`)**: Integrated `shap.Explainer` to calculate local feature attributions for every prediction.
- **Out-of-Distribution (OOD) Detection**: Built screening boundary checks flagging biometrics exceeding standard clinical bounds (e.g. Systolic BP > 240, HbA1c > 15%).
- **Calibrated Risk Bands**: Added `LOW`, `MODERATE`, and `HIGH` risk band badges.
- **Dual Patient/Clinician UI**: Enhanced `ResultCard.jsx` with Patient View (plain language drivers) vs Clinician View (SHAP attribution table).

#### Stage 5: Advanced ML Analytics & Registry
- **Model Registry Service (`server/src/modules/modelRegistry/`)**: Built `/api/v1/models` reporting ROC-AUC, PR-AUC, Brier score, Confusion Matrix, and Subgroup Fairness (sex/age).
- **Admin Model Analytics UI (`ModelAnalytics.jsx`)**: Built interactive metric inspection and champion model comparison page.

#### Stage 6: Evidence-Grounded AI / RAG Service
- **Guarded Medical RAG Service (`server/src/modules/knowledge/`)**: Built `/api/v1/knowledge/query` and `/documents` referencing ADA 2024, ACC/AHA 2019, WHO PEN 2023, and CDC 2023 guidelines.
- **Emergency Escalation Alerts**: Built acute symptom detection (chest pain, stroke, severe dyspnea) returning immediate 911 hotline alerts.
- **Medical Knowledge UI (`MedicalKnowledge.jsx`)**: Built searchable guideline library and verbatim citation cards.

#### Stage 7: OWASP ASVS L2 Security & Reliability
- **Helmet Security Headers**: Integrated `helmet()` middleware in Express API gateway.
- **Audit Logging Service (`server/src/modules/audit/`)**: Built append-only security audit trail recording user logins, registration, grants, and assessment events.
- **Admin Audit Log UI (`AuditLog.jsx`)**: Built searchable security event timeline.

#### Stage 8: MLOps & Input Drift Monitoring
- **Input Distribution Drift Detection (`ml/monitor.py`)**: Built Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistic calculators.
- **System Health & Drift Dashboard (`SystemHealth.jsx`)**: Built real-time service liveness and biometric feature drift monitoring page.
- **Platform Navigation**: Updated `Sidebar.jsx` with complete HealthLens AI navigation links.

---

## [3.1.0-stage0] - 2026-08-07

### Added - Stage 0 Roadmap & Security Baseline
- **HealthLens AI Transformation Plan**: Formally documented 24-week roadmap, ADR-001, and implementation plan.
- **Python ML Unit Test Suite**: Created `ml/test_extract.py` with 19 unit tests covering OCR parsing, name extraction, magic bytes detection, and file validation.
- **Environment Configuration**: Added `.env.example` templates at root, server, and client.
- **Custom Favicon**: Added custom heart-icon SVG favicon.

### Security & Safety Fixes
- **Patient Data Scrubbing**: Removed committed patient records containing real names from `server/data/db.json`. Replaced with empty history array.
- **Zero-Height Guard & Input Validation**: Added server and Python validation for positive `height_cm` to prevent HTTP 500 BMI division by zero.
- **OCR Review Gate**: Removed auto-trigger prediction after OCR document upload. Extracted fields populates form and displays review banner requiring explicit user submission.
- **Upload File Limits & Timeout**: Enforced 10 MB upload cap, MIME/magic-byte checks (PDF/JPEG/PNG), 30-second subprocess execution timeout, and guaranteed temporary file deletion.
- **CORS Whitelist**: Replaced open `cors()` with origin-domain check (`localhost:5173`, `localhost:5000`).
- **ChatBot Completion Lock**: Locked input and send actions permanently upon completing guided assessment.
- **Research PDF Language**: Replaced clinical diagnostic wording in PDF export with research screening disclaimers and stable session report IDs.
