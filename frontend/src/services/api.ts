import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for global error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  getInspectionLogs: (offset = 0, limit = 30) => 
    apiClient.get(`/api/inspection-logs`, { params: { offset, limit } }),
  
  overrideCamera: (data: any) => 
    apiClient.post(`/api/override`, data),
  
  reinspectLog: (logId: number) => 
    apiClient.post(`/api/reinspect-log/${logId}`),
    
  inspectImage: (formData: FormData) => 
    apiClient.post(`/api/inspect-image`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    }),
};
