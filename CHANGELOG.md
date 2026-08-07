# Changelog

All notable changes to the HealthLens AI project.

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

### Refactoring & Bug Fixes
- **Executable Python Repair**: Fixed `ml/utils.py` string escaping bug to ensure proper module imports.
- **Heart Model Contract**: Retained `id=0` placeholder in `build_heart_features` to preserve 15-feature shape for current model artifact (tagged for Stage 3 removal upon retraining).
- **ESLint Clean**: Resolved unused error parameter warnings in catch blocks via `_err` naming and `caughtErrorsIgnorePattern`.
- **Mobile Navigation**: Replaced static menu button in `Navbar.jsx` with responsive, accessible drawer navigation.
- **Directory Reorganization**: Moved sample files to `ml/test_fixtures/` and archived notebooks to `research/archive/notebooks/`.

---

## [3.0.0] - 2026-02-28

### Changed - Architecture & Tech Stack

- **Modernized Fullstack Application**:
  - Replaced legacy Streamlit frontend with React frontend powered by Vite.
  - Implemented Glassmorphism UI using Tailwind CSS v4.
  - Created dedicated Node.js/Express backend to serve RESTful API endpoints.
