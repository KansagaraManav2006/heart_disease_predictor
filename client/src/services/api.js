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
