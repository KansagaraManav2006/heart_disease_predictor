# PROJECT_AUDIT_REPORT.md

## HealthPredict — Security & Quality Audit Report

**Audit Date:** 2026-08-07  
**Branch:** Trupesh  
**Auditor:** Automated review + manual code inspection  
**Scope:** Full codebase audit of `heart_disease_predictor` repository  

---

> [!IMPORTANT]
> **This report documents software hardening of a research prototype.**
> It does not constitute clinical validation, regulatory certification, or approval for use with real patients.
> The application must not be described as a diagnostic tool or clinically approved system.

---

## Summary Statistics

| Category | Before | After |
|---|---|---|
| ESLint errors | 2 | **0** ✅ |
| Production build | Passing | **Passing** ✅ |
| Server npm high/critical | 2 high | **0** ✅ |
| Client npm high | 1 high + 4 moderate | **2 residual (see F-11)** |
| Committed patient records | 8 records with names | **0 (scrubbed)** ✅ |
| Unauthenticated history | All records exposed | **Empty session-only** ✅ |
| OCR auto-trigger | Yes (immediate predict) | **Blocked — user must review** ✅ |
| HTTP 500 on height=0 | Yes | **HTTP 400 with message** ✅ |
| Mobile menu functional | No | **Yes — accessible drawer** ✅ |
| Python tests | 1 placeholder | **19 tests, all passing** ✅ |

---

## Findings and Resolutions

### F-01 — ESLint Lint Failures

| Field | Detail |
|---|---|
| **Severity** | Medium |
| **Evidence** | `DiabetesPrediction.jsx:87`, `HeartDiseasePrediction.jsx:99` — `err` defined but unused in catch blocks |
| **Subsystem** | Frontend / CI |
| **Fix Applied** | Renamed `err` → `_err` in both catch blocks. Updated `eslint.config.js` to add `caughtErrorsIgnorePattern: '^_'` (standard convention for intentionally unused catch parameters) |
| **Verification** | `npx eslint .` exits 0, 0 errors |
| **Residual Risk** | None |

---

### F-02 — HTTP 500 on Zero/Missing Height (BMI Division by Zero)

| Field | Detail |
|---|---|
| **Severity** | High |
| **Evidence** | `utils.py:108` — `bmi_val = float(weight_kg) / ((float(height_cm) / 100.0) ** 2)` with no guard |
| **Subsystem** | ML / Python |
| **Fix Applied** | Added explicit guard in `build_heart_features()`: if `height_cm <= 0` raises `ValueError` before division. Server-side validation in `server/index.js` now rejects `height_cm=0` with HTTP 400 before spawning Python |
| **Verification** | Manual: POST `{ height_cm: 0 }` → HTTP 400 `{ error: "Invalid input.", details: [...] }` |
| **Residual Risk** | Low — client-side HTML `min="120"` also blocks zero in the browser |

---

### F-03 — OCR Auto-Triggers Prediction Without User Review

| Field | Detail |
|---|---|
| **Severity** | High |
| **Evidence** | `DiabetesPrediction.jsx:104`, `HeartDiseasePrediction.jsx:116` — `setTimeout(() => triggerPrediction(mergedData), 100)` immediately after OCR extract |
| **Subsystem** | Frontend |
| **Fix Applied** | `handleExtract()` in both pages now: (1) merges extracted data into form state, (2) switches to Manual tab, (3) clears any prior result/error, (4) sets `ocrExtracted = true` which shows a dismissible review banner. Prediction only fires when the user explicitly clicks Submit |
| **Verification** | Upload PDF → form populates but no prediction fires; blue/orange banner displayed |
| **Residual Risk** | None |

---

### F-04 — window.prompt() Usage in OCR Flow

| Field | Detail |
|---|---|
| **Severity** | Medium |
| **Evidence** | Both prediction pages used `window.prompt()` to ask for patient name when OCR failed to extract it |
| **Subsystem** | Frontend |
| **Fix Applied** | Removed all `window.prompt()` calls. Name is now a standard Optional form field that the user fills in the Manual Entry tab |
| **Verification** | Upload PDF without name field → form switches to manual tab; no browser dialog |
| **Residual Risk** | None |

---

### F-05 — Committed Patient Records with Real Names

| Field | Detail |
|---|---|
| **Severity** | Critical |
| **Evidence** | `server/data/db.json` contained 8 records with names "Harsh", "Manav", "Test Patient" and health data, committed to git history |
| **Subsystem** | Data / Privacy |
| **Fix Applied** | `server/data/db.json` replaced with `{ "history": [] }`. The server now uses session-only in-memory state (`let sessionHistory = []`) — no patient data is ever written to disk |
| **Verification** | `cat server/data/db.json` → `{ "history": [] }`. GET `/api/history` returns `[]` |
| **Residual Risk** | Medium — the original records remain in git history. To fully purge: run `git filter-repo --path server/data/db.json --invert-paths` and force-push. This is a deployment-team responsibility documented here |

---

### F-06 — Unauthenticated /api/history Exposes All Records

| Field | Detail |
|---|---|
| **Severity** | High |
| **Evidence** | `GET /api/history` returned every stored record to any caller with no auth check |
| **Subsystem** | Server / Auth |
| **Fix Applied** | History is now session-only in-memory (no persistence). Records are stored in `sessionHistory` array which lives only for the process lifetime. Phase 2 will add full authentication |
| **Verification** | Server restart → GET `/api/history` returns `[]` |
| **Residual Risk** | Medium — no auth yet (Phase 2 target). Current mitigation: nothing is persisted, so there is nothing to leak across sessions |

---

### F-07 — No Upload Size/Type Limits

| Field | Detail |
|---|---|
| **Severity** | High |
| **Evidence** | `multer({ dest: uploadDir })` with no `limits` or `fileFilter` — any file type and size accepted |
| **Subsystem** | Server |
| **Fix Applied** | Multer now configured with `limits: { fileSize: 10 * 1024 * 1024 }` (10 MB) and `fileFilter` that whitelists `application/pdf`, `image/jpeg`, `image/png`. Returns HTTP 413 on size and 415 on unsupported type. Python layer also validates magic bytes independently |
| **Verification** | Upload a .txt file → HTTP 415. Upload >10MB → HTTP 413 |
| **Residual Risk** | Low |

---

### F-08 — No Subprocess Timeout

| Field | Detail |
|---|---|
| **Severity** | Medium |
| **Evidence** | Python subprocesses for predict and extract had no timeout — a hung Python process would hold the request indefinitely |
| **Subsystem** | Server |
| **Fix Applied** | `runPython()` helper wraps all subprocesses with `setTimeout(30s)` + `proc.kill('SIGKILL')`. Guaranteed cleanup via async `cleanup()` helper that always deletes uploaded files regardless of outcome |
| **Verification** | Code review of `server/index.js:runPython()` |
| **Residual Risk** | Low |

---

### F-09 — Wide-Open CORS

| Field | Detail |
|---|---|
| **Severity** | Medium |
| **Evidence** | `app.use(cors())` with no origin restriction — accepts requests from any domain |
| **Subsystem** | Server |
| **Fix Applied** | CORS now uses an origin whitelist: `localhost:5173`, `localhost:5174`, `localhost:5000` in dev. Configurable via `ALLOWED_ORIGIN` env var for production |
| **Verification** | Request with `Origin: https://evil.com` → CORS error |
| **Residual Risk** | Low — production HTTPS and proper origin must be configured via env var at deployment |

---

### F-10 — Mobile Navigation Non-Functional

| Field | Detail |
|---|---|
| **Severity** | Medium |
| **Evidence** | `Navbar.jsx:26-30` — Menu button had no state or event handler; clicking did nothing |
| **Subsystem** | Frontend |
| **Fix Applied** | Added `menuOpen` state, toggle handler, backdrop overlay, accessible slide-in drawer nav with proper ARIA attributes (`aria-expanded`, `aria-controls`, `role="navigation"`, `aria-label`) |
| **Verification** | Mobile viewport: click Menu → drawer opens; click link → closes; click backdrop → closes |
| **Residual Risk** | None |

---

### F-11 — Vulnerable npm Dependencies

| Field | Detail |
|---|---|
| **Severity** | High (server), High (client) |
| **Evidence** | Server: express <4.22, multer <2.2, morgan <1.11. Client: jsPDF, react-router-dom, vite, postcss with advisories |
| **Subsystem** | Dependencies |
| **Fix Applied** | Server upgraded: express 4.22.2, multer 2.2.0, morgan 1.11.0 → **0 vulnerabilities**. Client upgraded: jspdf 4.2.1, react-router-dom 7.18.2, vite 7.3.6, postcss 8.5.26 |
| **Verification** | Server: `npm audit` → `found 0 vulnerabilities`. Client: 2 residual (see below) |
| **Residual Risk** | **Client — 2 high (GHSA-qwww-vcr4-c8h2):** React Router RSC CSRF bypass. **This advisory only affects apps using React Server Components + Server Actions.** This application is a pure client-side SPA with no RSC/server action usage. The vulnerability path does not exist in this codebase. The advisory's suggested fix (downgrade to 7.11.0) would itself be a breaking change with no security benefit here |

---

### F-12 — ChatBot Can Submit After Completion

| Field | Detail |
|---|---|
| **Severity** | Medium |
| **Evidence** | `ChatBot.jsx:118-123` — `disabled={currentStep >= questions.length}` on `<input>` but the keyboard `Enter` path and Send button path could still fire |
| **Subsystem** | Frontend |
| **Fix Applied** | Added `isComplete` boolean state. `handleSend()` guards on `isComplete` first. `handleKeyDown` also checks `isComplete`. Input shows "Assessment complete" placeholder when locked |
| **Verification** | Complete chat flow → input becomes read-only, Send button disabled, Enter key ignored |
| **Residual Risk** | None |

---

### F-13 — OCR File Type Detection by Extension Only

| Field | Detail |
|---|---|
| **Severity** | Medium |
| **Evidence** | `extract.py:13` — `ext = os.path.splitext(file_path)[1].lower()` used to select PDF vs image branch; multer-uploaded files have no extension |
| **Subsystem** | ML / Python |
| **Fix Applied** | `detect_file_type()` reads first 8 bytes and matches against magic byte signatures (PDF: `%PDF`, JPEG: `\xff\xd8\xff`, PNG: `\x89PNG\r\n\x1a\n`) |
| **Verification** | Python tests `TestDetectFileType` — all 4 cases pass |
| **Residual Risk** | None |

---

### F-14 — OCR Newline Replacement Breaks Name Extraction

| Field | Detail |
|---|---|
| **Severity** | Medium |
| **Evidence** | `extract.py:63` — `text = text.replace('\n', ' ')` destroyed line boundaries; patient name regex failed on multiline PDFs |
| **Subsystem** | ML / Python |
| **Fix Applied** | Replaced global newline removal with per-line stripping. `parse_medical_data()` now operates on both `lines` (for line-aware ops) and `single` (for cross-whitespace patterns). Name regex uses `re.MULTILINE` and a lookahead that stops at next field label |
| **Verification** | Python test `test_name_on_own_line` passes |
| **Residual Risk** | Low — OCR quality varies by document; complex layouts may still miss names |

---

### F-15 — raw_text Returned in OCR Response

| Field | Detail |
|---|---|
| **Severity** | Medium |
| **Evidence** | `extract.py:125` — raw OCR text (up to 500 chars) returned to client in API response |
| **Subsystem** | ML / Privacy |
| **Fix Applied** | `raw_text` removed from response. API now returns only: `extracted_data`, `confidence`, `warnings`, `missing_fields`, `detected_type` |
| **Verification** | Code review of `extract.py:main()` |
| **Residual Risk** | None |

---

### F-16 — Heart Feature Vector Includes Meaningless `id` Column

| Field | Detail |
|---|---|
| **Severity** | Medium |
| **Evidence** | `utils.py:125,147` — dataset row `id` included as a model feature; this is data leakage |
| **Subsystem** | ML |
| **Fix Applied** | `id` removed from `feature_row` dict and `expected_columns` list in `build_heart_features()` |
| **Verification** | Code review of `ml/utils.py` |
| **Residual Risk** | Medium — existing trained model artifacts were trained with `id` included. Model should be retrained (Phase 3) |

---

### F-17 — Circular Self-Reference Dependencies

| Field | Detail |
|---|---|
| **Severity** | Low |
| **Evidence** | Both `client/package.json` and `server/package.json` included `"disease-predictor-fullstack": "file:.."` as a production dependency pointing to their own parent |
| **Subsystem** | Build |
| **Fix Applied** | Removed from both `package.json` files |
| **Verification** | `npm install` completes without circular dependency warnings |
| **Residual Risk** | None |

---

### F-18 — Unused shadcn Dependency in Root

| Field | Detail |
|---|---|
| **Severity** | Low |
| **Evidence** | Root `package.json` listed `shadcn: ^3.8.5` as devDependency; not used anywhere in the codebase |
| **Subsystem** | Build |
| **Fix Applied** | Removed from `package.json` |
| **Verification** | Grep for shadcn usage returns no results |
| **Residual Risk** | None |

---

### F-19 — Missing / Incorrect ML Requirements

| Field | Detail |
|---|---|
| **Severity** | Low |
| **Evidence** | `requirements.txt` listed `fpdf` (not used) and was missing `PyMuPDF`, `pytesseract`, `Pillow` which are imported in `extract.py` |
| **Subsystem** | ML / Python |
| **Fix Applied** | Removed `fpdf`. Added `PyMuPDF>=1.24.0`, `pytesseract>=0.3.13`, `Pillow>=10.0.0` |
| **Verification** | `requirements.txt` matches actual imports in `extract.py` |
| **Residual Risk** | Low — Tesseract binary must be separately installed on host |

---

### F-20 — Default Vite Metadata and Wrong Version Claims

| Field | Detail |
|---|---|
| **Severity** | Low |
| **Evidence** | `index.html` title was `client`; favicon was Vite logo; `About.jsx` claimed React 18, Router v6 |
| **Subsystem** | Frontend |
| **Fix Applied** | Title updated to `HealthPredict — Risk Screening Tool`. Custom heart-icon favicon created. About page corrected: React 19, Vite 7, Tailwind CSS v4, Router v7 |
| **Verification** | Browser tab title correct; favicon visible |
| **Residual Risk** | None |

---

### F-21 — PDF Uses Diagnostic/Clinical Language

| Field | Detail |
|---|---|
| **Severity** | Medium |
| **Evidence** | PDF header: "Official Health Risk Assessment Report"; section: "Diagnostic Prediction"; random `Math.random()` Report ID |
| **Subsystem** | Frontend |
| **Fix Applied** | Header: "Risk Screening Summary (Research Use Only) — Not a clinical diagnosis". Section: "Risk Screening Result". Report ID: stable session-scoped counter. Footer disclaimer updated to explicitly state "FOR RESEARCH USE ONLY" |
| **Verification** | Click Download PDF → inspect PDF header and footer |
| **Residual Risk** | Low — patients may still misunderstand; further plain-language review recommended |

---

### F-22 — About Page Missing Privacy and Research Disclaimers

| Field | Detail |
|---|---|
| **Severity** | Low |
| **Evidence** | `About.jsx` made no mention of data handling, privacy, or research-use status |
| **Subsystem** | Frontend |
| **Fix Applied** | Added: (1) prominent amber research-use disclaimer banner at top of page; (2) privacy statement explaining session-only data, no external transmission, file deletion after OCR; (3) research disclaimer in footer |
| **Verification** | Navigate to `/about` — disclaimer banner visible |
| **Residual Risk** | None |

---

## Remaining Out-of-Scope Items (Phase 2 / Phase 3)

| Item | Phase |
|---|---|
| Authentication (register, login, sessions, CSRF, rate limiting) | Phase 2 |
| PostgreSQL persistent assessments with authorization | Phase 2 |
| RBAC (patient / clinician / admin roles) | Phase 2 |
| FastAPI internal ML service (model loaded once at startup) | Phase 3 |
| Docker Compose dev environment | Phase 3 |
| Reproducible ML training scripts (deterministic splits, calibration) | Phase 3 |
| Duplicate removal from diabetes training data | Phase 3 |
| Model retraining without `id` column | Phase 3 |
| Playwright E2E test suite | Phase 3 |
| `skops` model artifacts + metadata + model cards | Phase 3 |
| Git history purge of committed patient data | Deployment team |
| HTTPS termination and production secrets management | Deployment team |
| External clinical validation and regulatory certification | Out of scope |

---

## Disclaimer

This audit documents software quality and security hardening applied to a research prototype.
It does not constitute clinical validation under any regulatory framework (FDA, CE, MDR, or equivalent).
The application must not be described as a clinically approved or diagnostic tool.
Any use with real patients requires independent clinical validation, regulatory review, and legal compliance assessment beyond the scope of this work.
