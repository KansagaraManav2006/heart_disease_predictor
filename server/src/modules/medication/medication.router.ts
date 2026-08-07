import { Router, Request, Response, NextFunction } from 'express';
import { requireAuth } from '../../middleware/auth';

const router = Router();

// In-memory fallback dataset for medications
let mockMedications = [
  {
    id: 'med_1',
    name: 'Metformin HCl',
    dosage: '1000 mg',
    frequency: 'Twice daily with meals',
    category: 'Antidiabetic',
    status: 'ACTIVE',
    startDate: '2025-04-12',
    notes: 'Monitored for renal tolerance. eGFR must remain > 45 mL/min.',
    warning: null,
  },
  {
    id: 'med_2',
    name: 'Empagliflozin (Jardiance)',
    dosage: '10 mg',
    frequency: 'Once daily in morning',
    category: 'SGLT2 Inhibitor',
    status: 'ACTIVE',
    startDate: '2025-09-01',
    notes: 'Cardiorenal protective agent. Maintain hydration.',
    warning: null,
  },
  {
    id: 'med_3',
    name: 'Atorvastatin',
    dosage: '20 mg',
    frequency: 'Once daily at bedtime',
    category: 'Statin (Lipid Lowering)',
    status: 'ACTIVE',
    startDate: '2024-11-15',
    notes: 'Target LDL < 70 mg/dL.',
    warning: null,
  },
  {
    id: 'med_4',
    name: 'Lisinopril',
    dosage: '10 mg',
    frequency: 'Once daily',
    category: 'ACE Inhibitor (BP)',
    status: 'ACTIVE',
    startDate: '2025-01-20',
    notes: 'Monitored for serum potassium levels.',
    warning: 'Check serum potassium if combined with potassium supplements.',
  },
];

// GET /api/v1/medications
router.get('/', (_req: Request, res: Response) => {
  res.json({ medications: mockMedications });
});

// POST /api/v1/medications
router.post('/', (req: Request, res: Response) => {
  const { name, dosage, frequency, category, notes, warning } = req.body;
  if (!name || !dosage) {
    return res.status(400).json({ error: 'Medication name and dosage are required.' });
  }

  const newMed = {
    id: `med_${Date.now()}`,
    name,
    dosage,
    frequency: frequency || 'Once daily',
    category: category || 'General',
    status: 'ACTIVE',
    startDate: new Date().toISOString().split('T')[0],
    notes: notes || '',
    warning: warning || null,
  };

  mockMedications = [newMed, ...mockMedications];
  res.status(201).json({ medication: newMed });
});

// DELETE /api/v1/medications/:id
router.delete('/:id', (req: Request, res: Response) => {
  const { id } = req.params;
  mockMedications = mockMedications.filter((m) => m.id !== id);
  res.json({ success: true, deletedId: id });
});

export default router;
