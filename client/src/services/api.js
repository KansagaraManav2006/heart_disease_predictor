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
