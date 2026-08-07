import { Router, Request, Response } from 'express';

const router = Router();

let mockScenarios = [
  {
    id: 'sc_1',
    bmiReduction: 2.0,
    bpReduction: 10,
    glucoseReduction: 15,
    exerciseHours: 3.0,
    projectedRiskDelta: 38.0,
    createdAt: '2026-01-10T12:00:00Z',
  },
];

// GET /api/v1/risk-scenarios
router.get('/', (_req: Request, res: Response) => {
  res.json({ scenarios: mockScenarios });
});

// POST /api/v1/risk-scenarios
router.post('/', (req: Request, res: Response) => {
  const { bmiReduction, bpReduction, glucoseReduction, exerciseHours, projectedRiskDelta } = req.body;
  const newScenario = {
    id: `sc_${Date.now()}`,
    bmiReduction: Number(bmiReduction) || 0,
    bpReduction: Number(bpReduction) || 0,
    glucoseReduction: Number(glucoseReduction) || 0,
    exerciseHours: Number(exerciseHours) || 0,
    projectedRiskDelta: Number(projectedRiskDelta) || 0,
    createdAt: new Date().toISOString(),
  };

  mockScenarios.unshift(newScenario);
  res.status(201).json({ scenario: newScenario });
});

export default router;
