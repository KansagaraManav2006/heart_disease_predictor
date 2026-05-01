const express = require('express');
const cors = require('cors');
const path = require('path');
const morgan = require('morgan');
const { spawn } = require('child_process');
const fs = require('fs').promises;
const fsSync = require('fs');
const multer = require('multer');

const app = express();
const PORT = process.env.PORT || 5000;

// Set up temp upload directory
const uploadDir = path.join(__dirname, 'uploads');
if (!fsSync.existsSync(uploadDir)) {
  fsSync.mkdirSync(uploadDir);
}
const upload = multer({ dest: uploadDir });

// Set up data directory
const dataDir = path.join(__dirname, 'data');
if (!fsSync.existsSync(dataDir)) {
  fsSync.mkdirSync(dataDir);
  fsSync.writeFileSync(path.join(dataDir, 'db.json'), JSON.stringify({ history: [] }));
}

app.use(morgan('dev'));
app.use(cors());
app.use(express.json());

// Helper to execute the Python ML bridge script
function runPythonPrediction(type, data) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [
            path.join(__dirname, '../ml/predict.py'),
            type,
            JSON.stringify(data)
    ], {
      env: {
        ...process.env,
        OPENBLAS_NUM_THREADS: '1',
        OMP_NUM_THREADS: '1',
        MKL_NUM_THREADS: '1',
        NUMEXPR_NUM_THREADS: '1',
        OPENBLAS_CORETYPE: 'HASWELL',
      },
    });

        let outputData = '';
        let errorData = '';

        pythonProcess.stdout.on('data', (chunk) => {
            outputData += chunk.toString();
        });

        pythonProcess.stderr.on('data', (chunk) => {
            errorData += chunk.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                console.error(`Python script exited with code ${code}: ${errorData}`);
                return reject(new Error(errorData || 'Python script failed'));
            }
            try {
                // The python script should print exclusively the JSON string to stdout
                const result = JSON.parse(outputData.trim());
                if (result.error) {
                     console.error("Python ML script error:", result.error, result.traceback);
                     return reject(new Error(result.error));
                }
                resolve(result);
            } catch (err) {
                console.error("Failed to parse Python output. Raw output:", outputData);
                reject(new Error("Invalid JSON response from prediction model"));
            }
        });
    });
}

app.post('/api/predict/diabetes', async (req, res) => {
  try {
    console.log("Received diabetes assessment request:", req.body);
    const predictionResult = await runPythonPrediction('diabetes', req.body);
    res.json(predictionResult);
  } catch (err) {
    console.error("DIABETES PREDICTION CRASH:", err);
    res.status(500).json({ error: "Internal Server Error", details: err.message });
  }
});

app.post('/api/predict/heart', async (req, res) => {
  try {
    console.log("Received heart disease assessment request:", req.body);
    const predictionResult = await runPythonPrediction('heart', req.body);
    res.json(predictionResult);
  } catch (err) {
    console.error("HEART DISEASE PREDICTION CRASH:", err);
    res.status(500).json({ error: "Internal Server Error", details: err.message });
  }
});
// --- NEW ROUTES FOR OCR & HISTORY ---

app.post('/api/extract', upload.single('report'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  
  try {
    const filePath = req.file.path;
    
    // Spawn python script
    const pythonProcess = spawn('python', [
      path.join(__dirname, '../ml/extract.py'),
      filePath
    ], {
      env: {
        ...process.env,
        OPENBLAS_NUM_THREADS: '1',
        OMP_NUM_THREADS: '1',
        MKL_NUM_THREADS: '1',
        NUMEXPR_NUM_THREADS: '1',
        OPENBLAS_CORETYPE: 'HASWELL',
      },
    });

    let outputData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (chunk) => { outputData += chunk.toString(); });
    pythonProcess.stderr.on('data', (chunk) => { errorData += chunk.toString(); });

    pythonProcess.on('close', async (code) => {
      // Clean up uploaded file
      try { await fs.unlink(filePath); } catch(e) { console.error('Failed to clean up', e); }

      if (code !== 0) {
        console.error('OCR Python script failed:', errorData);
        return res.status(500).json({ error: 'OCR processing failed', details: errorData });
      }

      try {
        const result = JSON.parse(outputData.trim());
        res.json(result);
      } catch (err) {
        console.error('Failed to parse OCR output. Raw:', outputData);
        res.status(500).json({ error: 'Invalid response from OCR model' });
      }
    });
  } catch(err) {
    res.status(500).json({ error: "Internal Server Error", details: err.message });
  }
});

app.post('/api/history', async (req, res) => {
  try {
    const dbPath = path.join(__dirname, 'data', 'db.json');
    const rawData = await fs.readFile(dbPath, 'utf8');
    const db = JSON.parse(rawData);
    
    // Create new record
    const newRecord = {
      id: Date.now().toString(),
      date: new Date().toISOString(),
      ...req.body
    };
    
    db.history.push(newRecord);
    await fs.writeFile(dbPath, JSON.stringify(db, null, 2), 'utf8');
    res.json({ success: true, record: newRecord });
  } catch (err) {
    console.error('Failed to save to history:', err);
    res.status(500).json({ error: "Failed to save history" });
  }
});

app.get('/api/history', async (req, res) => {
  try {
    const dbPath = path.join(__dirname, 'data', 'db.json');
    if (!fsSync.existsSync(dbPath)) return res.json([]);
    const rawData = await fs.readFile(dbPath, 'utf8');
    const db = JSON.parse(rawData);
    res.json(db.history || []);
  } catch (err) {
    console.error('Failed to retrieve history:', err);
    res.status(500).json({ error: "Failed to load history" });
  }
});

// Serve frontend in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../client/dist')));

  app.get('*', (req, res) => {
    res.sendFile(path.resolve(__dirname, '../client', 'dist', 'index.html'));
  });
}

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
