import { APIError } from '../../middleware/errorHandler';

export interface MedicalCitation {
  id: string;
  sourceTitle: string;
  organization: string;
  publicationYear: number;
  section: string;
  url: string;
  snippet: string;
}

export interface KnowledgeQueryResponse {
  query: string;
  isEmergency: boolean;
  emergencyEscalationMessage?: string;
  answer: string;
  citations: MedicalCitation[];
  disclaimer: string;
}

const EMERGENCY_KEYWORDS = [
  'chest pain',
  'heart attack',
  'stroke',
  'shortness of breath',
  'severe dyspnea',
  'sudden numbness',
  'cardiac arrest',
  'unconscious',
  'severe pressure in chest',
];

const CURATED_KNOWLEDGE_BASE: MedicalCitation[] = [
  {
    id: 'kb_ada_2024_01',
    sourceTitle: 'Standards of Care in Diabetes—2024',
    organization: 'American Diabetes Association (ADA)',
    publicationYear: 2024,
    section: 'Classification and Diagnosis of Diabetes',
    url: 'https://diabetesjournals.org/care/issue/47/Supplement_1',
    snippet:
      'Diagnostic criteria for diabetes: Fasting plasma glucose >= 126 mg/dL (7.0 mmol/L), or 2-hour plasma glucose >= 200 mg/dL (11.1 mmol/L) during OGTT, or HbA1c >= 6.5% (48 mmol/mol), or in a patient with classic symptoms of hyperglycemia, a random plasma glucose >= 200 mg/dL.',
  },
  {
    id: 'kb_acc_aha_2019_01',
    sourceTitle: '2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease',
    organization: 'American College of Cardiology / American Heart Association',
    publicationYear: 2019,
    section: 'Cardiovascular Risk Assessment',
    url: 'https://www.ahajournals.org/doi/10.1161/CIR.0000000000000678',
    snippet:
      'Adults 40 to 75 years of age being evaluated for cardiovascular disease prevention should undergo 10-year atherosclerotic cardiovascular disease (ASCVD) risk estimation. Key risk factors include age, blood pressure, cholesterol levels, smoking status, and diabetes.',
  },
  {
    id: 'kb_who_2023_01',
    sourceTitle: 'WHO Package of Essential Noncommunicable Disease Interventions (WHO PEN)',
    organization: 'World Health Organization (WHO)',
    publicationYear: 2023,
    section: 'Cardiometabolic Risk Stratification',
    url: 'https://www.who.int/publications/i/item/9789240009226',
    snippet:
      'Total cardiovascular risk prediction charts evaluate 10-year risk of fatal or non-fatal cardiovascular events. Lifestyle interventions including daily physical activity (>=150 min/week), dietary sodium restriction (<2g/day), and tobacco cessation form the baseline recommendation for all risk tiers.',
  },
  {
    id: 'kb_cdc_diab_01',
    sourceTitle: 'National Diabetes Statistics & Screening Guidelines',
    organization: 'Centers for Disease Control and Prevention (CDC)',
    publicationYear: 2023,
    section: 'Prediabetes Risk Factors and Screening',
    url: 'https://www.cdc.gov/diabetes/basics/risk-factors.html',
    snippet:
      'Prediabetes is defined as HbA1c between 5.7% and 6.4%, or fasting blood glucose between 100 and 125 mg/dL. Early lifestyle modifications (5-7% weight reduction and regular moderate physical activity) reduce progression risk to type 2 diabetes by 58%.',
  },
];

export async function queryMedicalKnowledge(query: string): Promise<KnowledgeQueryResponse> {
  const normalizedQuery = query.trim().toLowerCase();

  // 1. Check Emergency Keywords
  const foundEmergency = EMERGENCY_KEYWORDS.find((kw) => normalizedQuery.includes(kw));
  if (foundEmergency) {
    return {
      query,
      isEmergency: true,
      emergencyEscalationMessage:
        'EMERGENCY ALERT: Your query mentions severe acute symptoms (' +
        foundEmergency +
        '). Please call emergency services (911 or your local emergency hotline) immediately or go to the nearest emergency department. Do not rely on an AI research tool for acute medical emergencies.',
      answer:
        'Immediate emergency escalation triggered due to detection of critical symptom keywords.',
      citations: [],
      disclaimer: 'EMERGENCY ESCALATION: Seek immediate emergency clinical care.',
    };
  }

  // 2. Keyword Search over Curated Knowledge Base
  const matchedCitations = CURATED_KNOWLEDGE_BASE.filter((doc) => {
    const text = (doc.sourceTitle + ' ' + doc.section + ' ' + doc.snippet).toLowerCase();
    const queryWords = normalizedQuery.split(/\s+/).filter((w) => w.length > 3);
    return queryWords.some((word) => text.includes(word));
  });

  const selectedCitations = matchedCitations.length > 0 ? matchedCitations : CURATED_KNOWLEDGE_BASE.slice(0, 2);

  // 3. Synthesize Evidence-Grounded Explanation
  let synthesizedAnswer =
    'Based on reviewed medical guidelines (' +
    selectedCitations.map((c) => c.organization).join(', ') +
    '), cardiometabolic risk screening relies on verified biomarker measurements including fasting glucose, HbA1c, blood pressure, lipid profiles, and lifestyle factors.';

  if (normalizedQuery.includes('glucose') || normalizedQuery.includes('hba1c') || normalizedQuery.includes('diabetes')) {
    synthesizedAnswer =
      'According to ADA 2024 and CDC screening guidelines, diabetes risk is evaluated using fasting glucose (>=126 mg/dL indicative of diabetes; 100-125 mg/dL for prediabetes) and HbA1c (>=6.5% indicative; 5.7-6.4% prediabetes). Regular screening and lifestyle intervention significantly mitigate progression.';
  } else if (normalizedQuery.includes('pressure') || normalizedQuery.includes('blood pressure') || normalizedQuery.includes('heart') || normalizedQuery.includes('hypertension')) {
    synthesizedAnswer =
      'Per ACC/AHA 2019 guidelines, blood pressure and total cholesterol categories are primary determinants of 10-year ASCVD event risk. Systolic pressure >130 mmHg or diastolic >80 mmHg warrants risk factor evaluation and lifestyle guidance.';
  }

  return {
    query,
    isEmergency: false,
    answer: synthesizedAnswer,
    citations: selectedCitations,
    disclaimer:
      'EVIDENCE-GROUNDED RESEARCH RESPONSE: Synthesized from peer-reviewed medical guidelines. For informational and educational research purposes only.',
  };
}

export async function getKnowledgeDocuments(): Promise<MedicalCitation[]> {
  return CURATED_KNOWLEDGE_BASE;
}
