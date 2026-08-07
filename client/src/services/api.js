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
