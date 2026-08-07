import { Router, Request, Response } from 'express';

const router = Router();

let mockTrendData: Array<{
  id: string;
  date: string;
  glucose: number;
  hba1c: number;
  systolic_bp: number;
  cholesterol: number;
  bmi: number;
}> = [
  { id: 'bt_1', date: '2025-03-01', glucose: 135, hba1c: 7.2, systolic_bp: 138, cholesterol: 215, bmi: 28.4 },
  { id: 'bt_2', date: '2025-06-15', glucose: 124, hba1c: 6.8, systolic_bp: 132, cholesterol: 198, bmi: 27.9 },
  { id: 'bt_3', date: '2025-09-20', glucose: 118, hba1c: 6.5, systolic_bp: 126, cholesterol: 188, bmi: 27.3 },
  { id: 'bt_4', date: '2026-01-10', glucose: 110, hba1c: 6.3, systolic_bp: 122, cholesterol: 178, bmi: 26.8 },
];

// GET /api/v1/biomarker-trends
router.get('/', (_req: Request, res: Response) => {
  res.json({ trends: mockTrendData });
});

// POST /api/v1/biomarker-trends
router.post('/', (req: Request, res: Response) => {
  const { glucose, hba1c, systolic_bp, cholesterol, bmi } = req.body;
  const newPoint = {
    id: `bt_${Date.now()}`,
    date: new Date().toISOString().split('T')[0],
    glucose: Number(glucose) || 0,
    hba1c: Number(hba1c) || 0,
    systolic_bp: Number(systolic_bp) || 0,
    cholesterol: Number(cholesterol) || 0,
    bmi: Number(bmi) || 0,
  };

  mockTrendData.push(newPoint);
  res.status(201).json({ point: newPoint });
});

export default router;
