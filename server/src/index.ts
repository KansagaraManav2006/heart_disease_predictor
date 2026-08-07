import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import cookieParser from 'cookie-parser';
import morgan from 'morgan';
import path from 'path';
import multer from 'multer';
import { spawn } from 'child_process';
import fs from 'fs';
import { promises as fsAsync } from 'fs';

import helmet from 'helmet';

import { ENV } from './config/env';
import v1Router from './routes/v1';
import { errorHandler, APIError } from './middleware/errorHandler';
import { globalApiRateLimiter } from './middleware/rateLimiter';

const app = express();

// ---------------------------------------------------------------------------
// Security & Middleware Stack
// ---------------------------------------------------------------------------
app.use(helmet());
app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) return callback(null, true);
      if (ENV.ALLOWED_ORIGINS.includes(origin)) return callback(null, true);
      callback(new Error(`CORS: Origin '${origin}' is not allowed.`));
    },
    credentials: true,
  })
);

app.use(morgan(ENV.IS_PROD ? 'combined' : 'dev'));
app.use(express.json({ limit: '1mb' }));
app.use(cookieParser());
app.use(globalApiRateLimiter);

// ---------------------------------------------------------------------------
// File Upload (Multer) Config
// ---------------------------------------------------------------------------
const ALLOWED_MIME_TYPES = new Set([
  'application/pdf',
  'image/jpeg',
  'image/png',
]);
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB

const uploadDir = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const upload = multer({
  dest: uploadDir,
  limits: { fileSize: MAX_UPLOAD_BYTES },
  fileFilter: (_req, file, cb) => {
    if (ALLOWED_MIME_TYPES.has(file.mimetype)) {
      cb(null, true);
    } else {
      cb(
        new Error(
          `Unsupported file type '${file.mimetype}'. Only PDF, JPEG, and PNG are accepted.`
        )
      );
    }
  },
});

// ---------------------------------------------------------------------------
// Python Subprocess Helper (30-second execution cap)
// ---------------------------------------------------------------------------
const SUBPROCESS_TIMEOUT_MS = 30_000;

function runPython(args: string[], env: Record<string, string> = {}): Promise<any> {
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

    const timer = setTimeout(() => {
      proc.kill('SIGKILL');
      reject(new Error('Python subprocess timed out after 30 seconds.'));
    }, SUBPROCESS_TIMEOUT_MS);

    proc.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    proc.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    proc.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        console.error(`Python exited ${code}: ${stderr}`);
        return reject(new Error(stderr || 'Python script execution failed.'));
      }
      try {
        const result = JSON.parse(stdout.trim());
        if (result.error) {
          console.error('Python reported error:', result.error);
          return reject(new Error(result.error));
        }
        resolve(result);
      } catch {
        console.error('Failed to parse Python JSON output. Raw:', stdout);
        reject(new Error('Invalid JSON response from ML service.'));
      }
    });

    proc.on('error', (err) => {
      clearTimeout(timer);
      reject(new Error(`Failed to spawn Python process: ${err.message}`));
    });
  });
}

// ---------------------------------------------------------------------------
// Heart Input Validation Helper
// ---------------------------------------------------------------------------
function validateHeartPayload(data: any): string[] {
  const errors: string[] = [];
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
// Route Mounts
// ---------------------------------------------------------------------------

// Versioned API v1 Router
app.use('/api/v1', v1Router);

// Session History (Session-only in-memory fallback for legacy UI)
const sessionHistory: any[] = [];

app.get('/api/history', (_req: Request, res: Response) => {
  res.json(sessionHistory);
});

app.post('/api/history', (req: Request, res: Response) => {
  const record = {
    id: `session_${Date.now()}`,
    date: new Date().toISOString(),
    ...req.body,
  };
  sessionHistory.push(record);
  res.json({ success: true, record });
});

// Legacy prediction and OCR bridges
app.post('/api/predict/diabetes', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const result = await runPython([
      path.join(__dirname, '../../ml/predict.py'),
      'diabetes',
      JSON.stringify(req.body),
    ]);
    res.json(result);
  } catch (err) {
    next(err);
  }
});

app.post('/api/predict/heart', async (req: Request, res: Response, next: NextFunction) => {
  const errors = validateHeartPayload(req.body);
  if (errors.length > 0) {
    return next(new APIError('Invalid input payload.', 400, errors));
  }
  try {
    const result = await runPython([
      path.join(__dirname, '../../ml/predict.py'),
      'heart',
      JSON.stringify(req.body),
    ]);
    res.json(result);
  } catch (err) {
    next(err);
  }
});

app.post('/api/extract', upload.single('report'), async (req: Request, res: Response, next: NextFunction) => {
  if (!req.file) {
    return next(new APIError('No file uploaded.', 400));
  }

  const filePath = req.file.path;
  const cleanup = async () => {
    try {
      await fsAsync.unlink(filePath);
    } catch {
      /* ignore */
    }
  };

  try {
    const result = await runPython([
      path.join(__dirname, '../../ml/extract.py'),
      filePath,
    ]);
    await cleanup();
    res.json(result);
  } catch (err) {
    await cleanup();
    next(err);
  }
});

// Production Static Client Serving
if (ENV.IS_PROD) {
  const clientDistDir = path.join(__dirname, '../../client/dist');
  app.use(express.static(clientDistDir));
  app.get('*', (_req: Request, res: Response) => {
    res.sendFile(path.join(clientDistDir, 'index.html'));
  });
}

// Global Error Handler
app.use(errorHandler);

// Start Server
app.listen(ENV.PORT, () => {
  console.log(`[HealthLens AI] Server listening on port ${ENV.PORT} (${ENV.NODE_ENV})`);
});
