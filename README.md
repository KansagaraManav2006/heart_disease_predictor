# Fullstack Disease Prediction System (Diabetes & Heart)

A modernized, clinical-grade web application for assessing diabetes and heart disease risk. The platform features a robust Express/Python backend engine and a professional medical dashboard frontend designed for healthcare environments.

## Core Features

- 🩺 **Dual Disease Diagnostics**: Comprehensive risk assessment for both Diabetes and Cardiovascular conditions.
- 🏥 **Clinical UI Design System**: A clean, professional medical dashboard layout (Medical Blue/Teal) built with React 19 and Tailwind CSS v4.
- 📄 **Automated Medical Reporting**: Instantly generate and download clinical-grade PDF Diagnostic Reports complete with patient metrics and physician signature lines (via `jsPDF`).
- 🚀 **Fullstack Architecture**: Complete separation of concerns between the Node.js Express API layer and the React client.
- 🎯 **Machine Learning Engine**: Trained scikit-learn predictive models dynamically executed via Python child processes.
- ⚡ **Rapid Development**: Configured with `concurrently` for instant one-command (`npm run dev`) full-stack startup.

## Architecture

The application has been upgraded from a monolithic Python Streamlit app to a modular MERN-inspired stack:

### `client/` (Frontend)
- **Framework**: React 18 built with Vite
- **Styling**: Tailwind CSS v4 using modern `@theme` CSS variables
- **Components**: Reusable, pure-functional components (`GlassCard`, `Button`, `ResultCard`)
- **Routing**: `react-router-dom` handling multi-page navigation

### `server/` (Backend)
- **Framework**: Node.js with Express
- **API**: RESTful endpoints (`/api/predict/diabetes`, `/api/predict/heart`) taking structured JSON payloads
- **CORS**: Configured for cross-origin local development

### `ml/` (Python Data Science Layer)
- `ml/utils.py` contains the feature engineering logic and artifact loading.
- Pretrained models and scalers (`ml/models/`) are loaded via Python. *(Note: Integration between Express and the Python models requires either an HTTP microservice like Flask or `child_process` spawning, depending on deployment environment.)*

## Setup and Development

1. **Install Root Dependencies**
   The root package manages concurrent execution:
   ```bash
   npm install
   npm run install:all  # Triggers npm install in both /client and /server
   ```

2. **Run the Development Server**
   Start both the Express backend and Vite frontend simultaneously:
   ```bash
   npm run dev
   ```
   - The React app will be available at `http://localhost:5173`
   - The Express API will be running on `http://localhost:5000`

## Production Build

To compile the React frontend for production distribution:

```bash
npm run build
```
The compiled assets will be placed in `client/dist/` and automatically served statically by the Express server when `NODE_ENV=production`. You can start the production server with:
```bash
npm start
```

## Deployment (Free Options)

This fullstack application with Python ML integration can be deployed using several free platforms:

### Option 1: Render (Recommended for Fullstack)

**Render** offers free tiers for web services and supports both Node.js and Python.

1. **Backend + Python ML Service**:
   - Deploy `server/` as a Node.js Web Service
   - Deploy `ml/` as a separate Python Web Service (Flask/FastAPI wrapper)
   - Link both services via environment variables
   - Free tier: 750 hours/month, auto-sleep after inactivity

2. **Frontend**:
   - Deploy `client/` as a Static Site
   - Connect to backend API URL
   - Free tier: Unlimited bandwidth

**Setup**:

```bash
# Create render.yaml in root for automated deployment
# Or use Render Dashboard to create services manually
```

### Option 2: Vercel (Frontend) + Railway (Backend + ML)

**Best for**: Separate frontend/backend deployment

- **Vercel** (Frontend):
  - Deploy React app from `client/` folder
  - 100GB bandwidth/month free
  - Automatic HTTPS and CDN

- **Railway** (Backend + ML):
  - Deploy Express server and Python ML service
  - $5 free credit monthly (~500 hours)
  - Connect MongoDB/PostgreSQL if needed

### Option 3: Fly.io (Fullstack + ML)

**Best for**: Dockerized deployments

- Deploy both Node.js backend and Python ML service
- Free tier: 3 shared-cpu VMs, 3GB storage
- Requires Docker configuration
- Excellent for production-grade deployment

### Option 4: Hugging Face Spaces (ML Model Only)

**Best for**: ML model serving

- Deploy Python ML models using Gradio/Streamlit interface
- Unlimited free CPU inference
- Community GPU available
- Perfect for ML experimentation

### Option 5: PythonAnywhere (Python ML Service)

**Best for**: Python-focused deployment

- Free tier: 1 web app, 512MB storage
- Deploy `ml/predict.py` as Flask API
- Combine with Vercel (frontend) + free Node backend elsewhere

### Recommended Stack for This Project

```text
Frontend (React):     Vercel or Netlify
Backend (Express):    Render or Railway
ML Service (Python):  Render or Railway
Database (if needed): MongoDB Atlas (Free 512MB)
```

### Pre-Deployment Checklist

- [ ] Set environment variables (API URLs, ports)
- [ ] Update CORS settings in `server/index.js`
- [ ] Install Python dependencies from `ml/requirements.txt`
- [ ] Test ML model loading and predictions
- [ ] Configure production build settings
- [ ] Set up CI/CD with GitHub Actions (optional)

## Legacy Streamlit

The initial version of this project used a Streamlit frontend (`app.py`). This architecture has been **deprecated and removed** in favor of the React/Node.js stack for greater scalability and UI control.

