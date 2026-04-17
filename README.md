# Health Disease Predictor

Full-stack healthcare risk assessment application for Diabetes and Heart Disease prediction.


This project combines:
- React + Vite frontend
- Node.js + Express backend
- Python machine learning inference (scikit-learn)
- Optional OCR extraction pipeline for lab report uploads

The application allows users to enter patient vitals, run predictive inference, view risk confidence, export reports, and store/retrieve assessment history.

## Project Overview

### What this system does
- Predicts Diabetes risk from clinical inputs.
- Predicts Heart Disease risk from clinical inputs.
- Shows prediction probability and risk level.
- Supports report upload with OCR-based value extraction.
- Stores prediction history in a lightweight JSON datastore.

### High-level architecture
1. Frontend submits form data to /api routes.
2. Express backend receives and validates payloads.
3. Backend spawns Python process (ml/predict.py) with payload.
4. Python loads trained model/scaler artifacts from ml/models.
5. Prediction response is returned to frontend as JSON.

## Repository Structure

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

## Core Features

- Dual disease prediction workflows.
- Clinical form UX with structured numeric/categorical inputs.
- PDF report generation on client side.
- OCR extraction from PDF/image reports (optional).
- Assessment history save and fetch endpoints.
- One-command dev startup for client and server.

## Technology Stack

### Frontend
- React 19
- Vite 7
- React Router 7
- Tailwind CSS 4
- jsPDF + jspdf-autotable

### Backend
- Node.js
- Express 4
- Multer (file uploads)
- Morgan (logging)
- CORS

### ML Layer
- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- fpdf

### OCR (used by ml/extract.py)
- PyMuPDF (fitz) for PDF text extraction
- Pillow + pytesseract for image OCR
- Tesseract OCR binary installed on OS PATH (system dependency)

## API Endpoints

### Prediction
- POST /api/predict/diabetes
- POST /api/predict/heart

### OCR
- POST /api/extract (multipart/form-data, field name: report)

### History
- POST /api/history
- GET /api/history

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

Run server:

npm start

When NODE_ENV=production, Express serves static files from client/dist.

## Important Implementation Notes

- Backend currently invokes Python using command name: python.
   Make sure python is available in terminal PATH, or update spawn command in server/index.js.

- History storage is file-based (server/data/db.json).
   This is suitable for local/small deployments and should be replaced with a database for larger scale.

- OCR endpoint accepts uploaded files, stores temporary files in server/uploads, and cleans them after processing.

## Common Issues and Fixes

### 1. Prediction fails with model loading errors
- Cause: Missing or wrong .pkl files in ml/models.
- Fix: Recreate or place correct artifacts with expected names.

### 2. Python process fails from backend
- Cause: python command not found or missing packages.
- Fix: Confirm Python PATH and install dependencies from ml/requirements.txt.

### 3. OCR extraction fails for images
- Cause: Tesseract not installed or not in PATH.
- Fix: Install Tesseract and restart terminal.

### 4. Port already in use
- Cause: Previous server process still running on 5000 or frontend on 5173.
- Fix: Stop old process or change ports.

## Scripts Reference

Root package.json scripts:
- npm run install:all  -> install server and client dependencies
- npm run client       -> run frontend only
- npm run server       -> run backend only
- npm run dev          -> run frontend and backend concurrently
- npm run build        -> build client
- npm start            -> start backend entry point

## Future Improvements

- Replace JSON history store with MongoDB/PostgreSQL.
- Add request validation and schema guards for all API payloads.
- Add unit and integration tests for Node and Python layers.
- Containerize full stack for reproducible deployment.
- Add CI pipeline for lint, tests, and build checks.

