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

// ---------------------------------------------------------------------------
// CORS — restrict to known dev origins; production should use env var.
// ---------------------------------------------------------------------------
const DEV_ORIGINS = [
  'http://localhost:5173',
  'http://localhost:5174',
  'http://localhost:5000',
];
const ALLOWED_ORIGIN =
  process.env.ALLOWED_ORIGIN
    ? process.env.ALLOWED_ORIGIN.split(',').map(s => s.trim())
    : DEV_ORIGINS;

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (curl, same-origin server-to-server)
    if (!origin) return callback(null, true);
    if (ALLOWED_ORIGIN.includes(origin)) return callback(null, true);
    callback(new Error(`CORS: Origin '${origin}' is not allowed.`));
  },
  credentials: true,
}));

app.use(morgan('dev'));
app.use(express.json({ limit: '1mb' }));

// ---------------------------------------------------------------------------
// Multer — whitelist PDF/JPEG/PNG by MIME type, hard-cap at 10 MB.
// ---------------------------------------------------------------------------
const ALLOWED_MIME_TYPES = new Set([
  'application/pdf',
  'image/jpeg',
  'image/png',
]);
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB

const uploadDir = path.join(__dirname, 'uploads');
if (!fsSync.existsSync(uploadDir)) fsSync.mkdirSync(uploadDir);

const upload = multer({
  dest: uploadDir,
  limits: { fileSize: MAX_UPLOAD_BYTES },
  fileFilter: (_req, file, cb) => {
    if (ALLOWED_MIME_TYPES.has(file.mimetype)) {
      cb(null, true);
    } else {
      cb(
        new Error(
          `Unsupported file type '${file.mimetype}'. ` +
          'Only PDF, JPEG, and PNG are accepted.'
        )
      );
    }
  },
});

// ---------------------------------------------------------------------------
// In-memory session history (never persisted — no personal data on disk).
// ---------------------------------------------------------------------------
let sessionHistory = [];

// ---------------------------------------------------------------------------
// Heart payload validation — guard against missing/zero height_cm.
// ---------------------------------------------------------------------------
function validateHeartPayload(data) {
  const errors = [];
  const height = Number(data.height_cm ?? data.height ?? 0);
  const weight = Number(data.weight_kg ?? data.weight ?? 0);
  const systolic = Number(data.systolic_bp ?? data.systolic ?? 0);
  const diastolic = Number(data.diastolic_bp ?? data.diastolic ?? 0);

  if (!Number.isFinite(height) || height <= 0) {
    errors.push('height_cm must be a positive number (e.g. 170).');
  }
  if (!Number.isFinite(weight) || weight <= 0) {
    errors.push('weight_kg must be a positive number (e.g. 70).');
  }
  if (!Number.isFinite(systolic) || systolic <= 0) {
    errors.push('systolic_bp must be a positive number (e.g. 120).');
  }
  if (!Number.isFinite(diastolic) || diastolic <= 0) {
    errors.push('diastolic_bp must be a positive number (e.g. 80).');
  }
  return errors;
}

// ---------------------------------------------------------------------------
// Helper: run Python subprocess with a 30-second timeout.
// ---------------------------------------------------------------------------
const SUBPROCESS_TIMEOUT_MS = 30_000;

function runPython(args, env = {}) {
  return new Promise((resolve, reject) => {
    const childEnv = {
      ...process.env,
      OPENBLAS_NUM_THREADS: '1',
      OMP_NUM_THREADS: '1',
      MKL_NUM_THREADS: '1',
      NUMEXPR_NUM_THREADS: '1',
      OPENBLAS_CORETYPE: 'HASWELL',
      ...env,
    };

    const proc = spawn('python', args, { env: childEnv });

    let stdout = '';
    let stderr = '';

    // Hard timeout: kill the process if it takes too long.
    const timer = setTimeout(() => {
      proc.kill('SIGKILL');
      reject(new Error('Python subprocess timed out after 30 seconds.'));
    }, SUBPROCESS_TIMEOUT_MS);

    proc.stdout.on('data', chunk => { stdout += chunk.toString(); });
    proc.stderr.on('data', chunk => { stderr += chunk.toString(); });

    proc.on('close', code => {
      clearTimeout(timer);
      if (code !== 0) {
        console.error(`Python exited ${code}: ${stderr}`);
        return reject(new Error(stderr || 'Python script failed'));
      }
      try {
        const result = JSON.parse(stdout.trim());
        if (result.error) {
          console.error('Python script reported error:', result.error);
          return reject(new Error(result.error));
        }
        resolve(result);
      } catch (parseErr) {
        console.error('Failed to parse Python output. Raw:', stdout);
        reject(new Error('Invalid JSON response from ML service.'));
      }
    });

    proc.on('error', err => {
      clearTimeout(timer);
      reject(new Error(`Failed to spawn Python: ${err.message}`));
    });
  });
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

// Liveness / health check
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Diabetes prediction
app.post('/api/predict/diabetes', async (req, res) => {
  try {
    console.log('Received diabetes assessment request');
    const result = await runPython([
      path.join(__dirname, '../ml/predict.py'),
      'diabetes',
      JSON.stringify(req.body),
    ]);
    res.json(result);
  } catch (err) {
    console.error('DIABETES PREDICTION ERROR:', err.message);
    res.status(500).json({ error: 'Prediction failed.', details: err.message });
  }
});

// Heart disease prediction — validate payload first
app.post('/api/predict/heart', async (req, res) => {
  const errors = validateHeartPayload(req.body);
  if (errors.length > 0) {
    return res.status(400).json({ error: 'Invalid input.', details: errors });
  }
  try {
    console.log('Received heart disease assessment request');
    const result = await runPython([
      path.join(__dirname, '../ml/predict.py'),
      'heart',
      JSON.stringify(req.body),
    ]);
    res.json(result);
  } catch (err) {
    console.error('HEART PREDICTION ERROR:', err.message);
    res.status(500).json({ error: 'Prediction failed.', details: err.message });
  }
});

// OCR extraction — type-restricted upload, timeout-guarded subprocess
app.post('/api/extract', upload.single('report'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }

  const filePath = req.file.path;

  // Cleanup helper — always remove the temp file regardless of outcome.
  const cleanup = async () => {
    try { await fs.unlink(filePath); } catch { /* ignore */ }
  };

  try {
    const result = await runPython([
      path.join(__dirname, '../ml/extract.py'),
      filePath,
    ]);
    await cleanup();
    res.json(result);
  } catch (err) {
    await cleanup();
    console.error('OCR ERROR:', err.message);
    res.status(500).json({ error: 'OCR processing failed.', details: err.message });
  }
});

// Session-scoped history — no persistence, no personal data on disk
app.get('/api/history', (_req, res) => {
  res.json(sessionHistory);
});

app.post('/api/history', (req, res) => {
  const record = {
    id: `session_${Date.now()}`,
    date: new Date().toISOString(),
    ...req.body,
  };
  sessionHistory.push(record);
  res.json({ success: true, record });
});

// Serve frontend in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../client/dist')));
  app.get('*', (_req, res) => {
    res.sendFile(path.resolve(__dirname, '../client', 'dist', 'index.html'));
  });
}

// Global error handler for multer size/type errors
app.use((err, _req, res, _next) => {
  if (err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({ error: 'File too large. Maximum size is 10 MB.' });
  }
  if (err.message?.startsWith('Unsupported file type')) {
    return res.status(415).json({ error: err.message });
  }
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error.' });
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
