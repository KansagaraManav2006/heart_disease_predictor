# Fullstack Disease Prediction System (v2.0)

A modernized, clinical-grade web application for assessing diabetes and heart disease risk. The platform features a robust Express/Python backend engine, an AI-driven Chat Assistant, Automated OCR Report extraction, and a professional medical dashboard designed for healthcare environments.

## Repository Structure

```text
heart_disease_predictor-main/
|- client/                  React app (UI)
|  |- src/components/       Reusable UI components
|  |- src/pages/            Screen-level pages
|  |- src/services/api.js   API client methods
|- server/                  Express API
|  |- index.js              Routes + Python bridge + history + OCR upload
|  |- data/db.json          Local history storage
|- ml/                      Python inference and data utilities
|  |- predict.py            Entry point for predictions
|  |- extract.py            OCR extraction logic
|  |- utils.py              Feature engineering + artifact loading
|  |- models/               Trained model/scaler artifacts (.pkl)
|  |- requirements.txt      Python dependencies
|- README.md
|- DOCUMENTATION.md
```

---

## 🚀 Core Features

- 🩺 **Dual Disease Diagnostics**: Comprehensive risk assessment for both Diabetes and Cardiovascular conditions utilizing Pre-Trained Machine Learning Models (Random Forest & Logistic Regression).
- 🏥 **Clinical UI Design System**: A clean, professional medical dashboard layout built with React 18 and Tailwind CSS v4, optimized for clinical trust and readability.
- 📄 **Automated Medical Reporting**: Instantly generate and download clinical-grade PDF Diagnostic Reports complete with patient metrics and actionable suggestions (via `jsPDF`).
- 📸 **Automated OCR Upload**: Upload laboratory PDFs or images (via `Tesseract OCR` and `PyMuPDF`) to automatically extract vital signs like Blood Pressure, Glucose, and BMI to fill out the diagnostic forms instantly.
- 💬 **AI Chat Assistant**: A conversational user interface that guides patients step-by-step to input their biological metrics without the stress of navigating large clinical forms.
- 📈 **Longitudinal Dashboard**: A dedicated dashboard to track a patient's historical scans, comparing their current health metrics to their previous tests (e.g. tracking improving blood pressure).
- 💡 **Suggestion Engine**: Rule-based logic evaluating user metrics to provide tailored, actionable lifestyle advice.

---

## 🏗️ System Architecture

The application operates on a powerful, split-layer MERN-inspired stack:

### `client/` (Frontend)
- **Framework**: React 18 built with Vite for sub-second hot module replacement.
- **Styling**: Tailwind CSS v4.
- **Components**: Reusable, pure-functional components (`GlassCard`, `Button`, `ResultCard`, `ChatBot`).
- **Routing**: `react-router-dom` handling multi-page navigation.

### `server/` (Backend)
- **Framework**: Node.js with Express.
- **API**: RESTful endpoints (`/api/predict/diabetes`, `/api/predict/heart`, `/api/history`, `/api/extract`) taking structured JSON payloads.
- **Persistence**: Lightweight JSON-based local database (`db.json`) tied to browser-generated `userId`.
- **Bridge**: Utilizes `child_process.spawn` to instantiate Python instances asynchronously on-demand.

### `ml/` (Python Data Science Layer)
- **Engine**: Python 3.10+
- **Libraries**: `pandas`, `scikit-learn`, `pytesseract`, `PyMuPDF`, `numpy`.
- **Workflow**: `predict.py` executes models stored in `.pkl` format, taking sanitized CLI arguments from Node.js and outputting the final diagnostic probabilities.

*(See `DOCUMENTATION.md` for an extremely detailed deep-dive into the architectural decisions, machine learning algorithms, and engineering challenges.)*

---

## 💻 Setup and Local Development

### 1. Prerequisites
- **Node.js** (v18+)
- **Python** (v3.9+)
- **Tesseract OCR**: You *must* have the Tesseract binary installed on your host system to use the OCR upload feature.
   - Windows: Install via [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.
   - Mac: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

### 2. Install Dependencies
```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Install the Python dependencies
cd ml
pip install -r requirements.txt
cd ..

# Install the Node/React dependencies
npm install
npm run install:all  # Triggers npm install in both /client and /server
```

### 3. Run the Development Server
Start both the Express backend and Vite frontend simultaneously:
```bash
npm run dev
```
- The React App will be running at `http://localhost:5173`
- The Express API will be running on `http://localhost:5000`

---

## 📦 Production Build

## Prerequisites

Install these before running the app:
- Node.js 18+
- npm 9+
- Python 3.10+

Optional for OCR image extraction:
- Tesseract OCR installed and available in PATH

## Setup and Run (Development)

### 1. Install Node dependencies

From project root:

npm install
npm run install:all

### 2. Install Python dependencies

From project root:

pip install -r ml/requirements.txt

If you want OCR support, also install:

pip install pymupdf pillow pytesseract

### 3. Verify ML artifacts exist

Ensure these files are present under ml/models:
- diabetes_model.pkl
- diabetes_scaler.pkl
- heart_model.pkl
- heart_scaler.pkl

### 4. Start full stack

From project root:

npm run dev

Default local URLs:
- Frontend: http://localhost:5173
- Backend: http://localhost:5000

Note: Vite proxy forwards /api requests to backend port 5000.

## Production Commands

Build frontend bundle:

npm run build
```bash
npm run build
```
The compiled assets will be placed in `client/dist/`. 

---

## 🌐 Deployment Guidelines

Because this application relies on a Node.js server interacting directly with a Python execution layer, you must deploy the backend to a service that supports custom environments.

### Recommended Production Stack
- **Frontend**: Vercel or Netlify (Deploy `client/`)
- **Backend & ML**: Render or Railway (Deploy `server/` and `ml/`)

**Backend Deployment Steps (Render/Railway):**
1. Ensure the platform supports installing both Node.js and Python.
2. In your build command, you must install both environments:
   ```bash
   npm install && cd ml && pip install -r requirements.txt && cd ..
   ```
3. Set the start command to boot the Express server: `npm start`
4. Make sure Tesseract OCR is available in the deployment container (via `apt-get` in a Dockerfile or Buildpack).

---

## 🔧 Common Issues and Fixes

### 1. Prediction fails with model loading errors
- **Cause:** Missing or wrong `.pkl` files in `ml/models`.
- **Fix:** Recreate or place correct artifacts with expected names.

### 2. Python process fails from backend
- **Cause:** `python` command not found or missing packages.
- **Fix:** Confirm Python PATH and install dependencies from `ml/requirements.txt`.

### 3. OCR extraction fails for images
- **Cause:** Tesseract not installed or not in PATH.
- **Fix:** Install Tesseract and restart terminal.

### 4. Port already in use
- **Cause:** Previous server process still running on 5000 or frontend on 5173.
- **Fix:** Stop old process or change ports.

---

## 📄 Detailed Documentation

For a highly detailed technical breakdown answering **"What we use and exactly why we use it"**, covering the supervised learning methodologies, data engineering challenges, the suggestion engine, and the IPC bridging mechanics, please see the enclosed [`DOCUMENTATION.md`](./DOCUMENTATION.md) file.
