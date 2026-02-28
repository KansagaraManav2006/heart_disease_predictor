const express = require('express');
const cors = require('cors');
const path = require('path');
const morgan = require('morgan');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 5000;

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
        ]);

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
