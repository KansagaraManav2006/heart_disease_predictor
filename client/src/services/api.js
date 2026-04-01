// The Vite proxy handles routing /api to the Express server running on port 5000
const API_BASE_URL = '/api';

export const predictDiabetes = async (patientData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict/diabetes`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(patientData),
    });
    
    if (!response.ok) {
      throw new Error('Failed to get prediction');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error predicting diabetes:', error);
    throw error;
  }
};

export const predictHeartDisease = async (patientData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict/heart`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(patientData),
    });
    
    if (!response.ok) {
      throw new Error('Failed to get prediction');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error predicting heart disease:', error);
    throw error;
  }
};

export const uploadReport = async (file) => {
  try {
    const formData = new FormData();
    formData.append('report', file);

    const response = await fetch(`${API_BASE_URL}/extract`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) throw new Error('Failed to extract data');
    return await response.json();
  } catch (error) {
    console.error('OCR Extraction error:', error);
    throw error;
  }
};

export const getHistory = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/history`);
    if (!response.ok) throw new Error('Failed to fetch history');
    return await response.json();
  } catch (error) {
    console.error('History fetch error:', error);
    throw error;
  }
};

export const saveHistory = async (record) => {
  try {
    const response = await fetch(`${API_BASE_URL}/history`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(record),
    });
    if (!response.ok) throw new Error('Failed to save history');
    return await response.json();
  } catch (error) {
    console.error('Save history error:', error);
    throw error;
  }
};
