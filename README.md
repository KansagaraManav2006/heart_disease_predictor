# Fullstack Disease Prediction System (Diabetes & Heart)

A modernized web application for predicting diabetes and heart disease risk. The application features a robust Express backend API and a cutting-edge React frontend adorned with a Glassmorphism aesthetic.

## Features

- 🩺 **Dual Disease Prediction**: Diabetes and Heart Disease risk assessment
- 🔮 **Glassmorphism UI**: Beautiful, translucent interface powered by Tailwind CSS v4 and React 18
- 🚀 **Fullstack Architecture**: Clean separation between the Node.js API layer and the React client
- 🎯 **Accurate Models**: Trained scikit-learn models powering the data science layer
- ⚡ **Rapid Development**: Configured with `concurrently` for one-command frontend and backend startup

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

## Legacy Streamlit
The initial version of this project used a Streamlit frontend (`app.py`). This architecture has been **deprecated and removed** in favor of the React/Node.js stack for greater scalability and UI control.

## Credits
Built by Smit Kansagara
