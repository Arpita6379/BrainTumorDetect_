const API_URL = 'http://localhost:5000/api';

// User authentication services
export const register = async (name, email, password) => {
  try {
    const response = await fetch(`${API_URL}/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name, email, password }),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Registration failed');
    }
    
    return data;
  } catch (error) {
    throw error;
  }
};

export const login = async (email, password) => {
  try {
    const response = await fetch(`${API_URL}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
      credentials: 'include',
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Login failed');
    }
    
    // Save user to local storage
    localStorage.setItem('user', JSON.stringify(data.user));
    
    return data;
  } catch (error) {
    throw error;
  }
};

export const logout = () => {
  // Remove user from local storage
  localStorage.removeItem('user');
};

export const getCurrentUser = () => {
  const userStr = localStorage.getItem('user');
  if (!userStr) return null;
  return JSON.parse(userStr);
};

// Image processing services
export const uploadImage = async (file, userId) => {
  try {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('user_id', userId);
    
    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Upload failed');
    }
    
    return data;
  } catch (error) {
    throw error;
  }
};

export const getHistory = async (userId) => {
  try {
    const response = await fetch(`${API_URL}/history/${userId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to fetch history');
    }
    
    return data.history;
  } catch (error) {
    throw error;
  }
};

export const getImage = async (imageId) => {
  try {
    const response = await fetch(`${API_URL}/image/${imageId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to fetch image');
    }
    
    return data;
  } catch (error) {
    throw error;
  }
};

// Starred images services
export const getStarredImages = async (userId) => {
  try {
    const response = await fetch(`${API_URL}/starred/${userId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to fetch starred images');
    }
    
    return data.starred_images || {};
  } catch (error) {
    console.error('Error fetching starred images:', error);
    throw error;
  }
};

export const addStarredImage = async (userId, imageId, note) => {
  try {
    const response = await fetch(`${API_URL}/starred`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        image_id: imageId,
        note: note
      }),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to star image');
    }
    
    return data;
  } catch (error) {
    throw error;
  }
};

export const removeStarredImage = async (userId, imageId) => {
  try {
    const response = await fetch(`${API_URL}/starred/${userId}/${imageId}`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to remove starred image');
    }
    
    return data;
  } catch (error) {
    throw error;
  }
};

// Debug API
export const debugDatabase = async (userId) => {
  try {
    const response = await fetch(`${API_URL}/debug/${userId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to get debug info');
    }
    
    return data;
  } catch (error) {
    console.error('Debug API error:', error);
    throw error;
  }
}; 