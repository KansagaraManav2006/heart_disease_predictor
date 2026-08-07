const API_BASE_URL = '/api';

// CSRF token storage
let currentCSRFToken = '';

export const getCSRFToken = async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/v1/auth/csrf`, {
      credentials: 'include',
    });
    if (res.ok) {
      const data = await res.json();
      currentCSRFToken = data.csrfToken || '';
      return currentCSRFToken;
    }
  } catch (err) {
    console.error('Failed to fetch CSRF token:', err);
  }
  return '';
};

const authenticatedFetch = async (url, options = {}) => {
  if (!currentCSRFToken && ['POST', 'PUT', 'DELETE', 'PATCH'].includes(options.method?.toUpperCase())) {
    await getCSRFToken();
  }

  const headers = {
    ...options.headers,
    ...(currentCSRFToken && { 'x-csrf-token': currentCSRFToken }),
  };

  const response = await fetch(url, {
    ...options,
    headers,
    credentials: 'include',
  });

  return response;
};

// ---------------------------------------------------------------------------
// Authentication API (v1)
// ---------------------------------------------------------------------------

export const register = async ({ email, password, role, fullName }) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, role, fullName }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || 'Registration failed');
  }
  return data;
};

export const login = async ({ email, password }) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || 'Login failed');
  }
  if (data.csrfToken) {
    currentCSRFToken = data.csrfToken;
  }
  return data;
};

export const logout = async () => {
  try {
    await authenticatedFetch(`${API_BASE_URL}/v1/auth/logout`, {
      method: 'POST',
    });
  } finally {
    currentCSRFToken = '';
  }
};

export const fetchMe = async () => {
  const response = await fetch(`${API_BASE_URL}/v1/auth/me`, {
    credentials: 'include',
  });
  if (response.status === 401) {
    return null;
  }
  if (!response.ok) {
    throw new Error('Failed to fetch user session');
  }
  const data = await response.json();
  return data.user;
};

export const verifyEmail = async (token) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/auth/verify-email`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || 'Email verification failed');
  }
  return data;
};

// ---------------------------------------------------------------------------
// Patient & Access Grants API (v1)
// ---------------------------------------------------------------------------

export const getMyProfile = async () => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/patient/me`);
  if (!response.ok) throw new Error('Failed to fetch patient profile');
  const data = await response.json();
  return data.profile;
};

export const updateMyProfile = async (profileData) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/patient/me`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profileData),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to update profile');
  return data.profile;
};

export const grantClinicianAccess = async (clinicianEmail) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/access/grants`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clinicianEmail }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to grant clinician access');
  return data.grant;
};

export const revokeClinicianAccess = async (grantId) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/access/grants/${grantId}`, {
    method: 'DELETE',
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to revoke access');
  return data.grant;
};

export const getAssignedPatients = async () => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/access/assigned-patients`);
  if (!response.ok) throw new Error('Failed to fetch assigned patients');
  const data = await response.json();
  return data.patients;
};

// ---------------------------------------------------------------------------
// System Health & Drift API (v1)
// ---------------------------------------------------------------------------

export const getSystemHealth = async () => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/system-health`);
  if (!response.ok) throw new Error('Failed to fetch system health & drift report');
  return await response.json();
};

// ---------------------------------------------------------------------------
// Audit Events API (v1)
// ---------------------------------------------------------------------------

export const getAuditEvents = async () => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/audit`);
  if (!response.ok) throw new Error('Failed to fetch audit trail events');
  return await response.json();
};

// ---------------------------------------------------------------------------
// Knowledge & Guidelines RAG API (v1)
// ---------------------------------------------------------------------------

export const queryKnowledge = async (query) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/knowledge/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to query medical knowledge');
  return data;
};

export const getKnowledgeDocuments = async () => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/knowledge/documents`);
  if (!response.ok) throw new Error('Failed to fetch guideline documents');
  const data = await response.json();
  return data.documents;
};

// ---------------------------------------------------------------------------
// Model Registry API (v1)
// ---------------------------------------------------------------------------

export const getModels = async () => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/models`);
  if (!response.ok) throw new Error('Failed to fetch model registry list');
  const data = await response.json();
  return data.models;
};

export const getCurrentModel = async (condition) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/models/${condition}/current`);
  if (!response.ok) throw new Error('Failed to fetch current model metadata');
  const data = await response.json();
  return data.model;
};

// ---------------------------------------------------------------------------
// Assessments API (v1)
// ---------------------------------------------------------------------------

export const recordAssessment = async (assessmentData) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/assessments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(assessmentData),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Failed to record assessment');
  return data.assessment;
};

export const getMyAssessments = async (condition) => {
  const url = condition
    ? `${API_BASE_URL}/v1/assessments?condition=${condition}`
    : `${API_BASE_URL}/v1/assessments`;
  const response = await authenticatedFetch(url);
  if (!response.ok) throw new Error('Failed to fetch assessments');
  const data = await response.json();
  return data.assessments;
};

export const getAssessmentById = async (id) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/v1/assessments/${id}`);
  if (!response.ok) throw new Error('Failed to fetch assessment details');
  const data = await response.json();
  return data.assessment;
};

// ---------------------------------------------------------------------------
// Predictions & OCR
// ---------------------------------------------------------------------------

export const predictDiabetes = async (patientData) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/predict/diabetes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patientData),
  });
  if (!response.ok) throw new Error('Failed to get prediction');
  return await response.json();
};

export const predictHeartDisease = async (patientData) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/predict/heart`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patientData),
  });
  if (!response.ok) throw new Error('Failed to get prediction');
  return await response.json();
};

export const uploadReport = async (file) => {
  const formData = new FormData();
  formData.append('report', file);

  const response = await authenticatedFetch(`${API_BASE_URL}/extract`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) throw new Error('Failed to extract data');
  return await response.json();
};

export const getHistory = async () => {
  const response = await authenticatedFetch(`${API_BASE_URL}/history`);
  if (!response.ok) throw new Error('Failed to fetch history');
  return await response.json();
};

export const saveHistory = async (record) => {
  const response = await authenticatedFetch(`${API_BASE_URL}/history`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(record),
  });
  if (!response.ok) throw new Error('Failed to save history');
  return await response.json();
};

// ---------------------------------------------------------------------------
// Guided Chatbot Helpers
// ---------------------------------------------------------------------------

export const getBotQuestions = async (condition) => {
  if (condition === 'diabetes') {
    return [
      { key: 'pregnancies', text: 'How many pregnancies have you had? (Enter 0 if not applicable)', label: 'Pregnancies', unit: 'count' },
      { key: 'glucose', text: 'What is your fasting blood glucose level in mg/dL?', label: 'Fasting Glucose', unit: 'mg/dL' },
      { key: 'bloodPressure', text: 'What is your diastolic blood pressure in mmHg?', label: 'Diastolic BP', unit: 'mmHg' },
      { key: 'skinThickness', text: 'What is your triceps skin fold thickness in mm?', label: 'Skin Thickness', unit: 'mm' },
      { key: 'insulin', text: 'What is your 2-Hour serum insulin level in mu U/ml?', label: '2-Hour Insulin', unit: 'mu U/ml' },
      { key: 'bmi', text: 'What is your Body Mass Index (BMI) in kg/m²?', label: 'BMI', unit: 'kg/m²' },
      { key: 'dpf', text: 'What is your Diabetes Pedigree Function score (0.05 to 2.5)?', label: 'Diabetes Pedigree Function', unit: 'score' },
      { key: 'age', text: 'What is your age in years?', label: 'Age', unit: 'years' },
    ];
  } else {
    return [
      { key: 'age', text: 'What is your age in years?', label: 'Age', unit: 'years' },
      { key: 'systolic_bp', text: 'What is your resting systolic blood pressure in mmHg?', label: 'Systolic BP', unit: 'mmHg' },
      { key: 'cholesterol', text: 'What is your serum cholesterol in mg/dL?', label: 'Cholesterol', unit: 'mg/dL' },
      { key: 'max_heart_rate', text: 'What is your maximum heart rate achieved?', label: 'Max Heart Rate', unit: 'bpm' },
      { key: 'st_depression', text: 'What is your ST depression induced by exercise relative to rest?', label: 'ST Depression', unit: 'mm' },
    ];
  }
};

export const processBotAnswers = async (condition, answers) => {
  if (condition === 'diabetes') {
    return await predictDiabetes(answers);
  } else {
    return await predictHeartDisease(answers);
  }
};
