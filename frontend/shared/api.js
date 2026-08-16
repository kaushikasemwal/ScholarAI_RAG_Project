/**
 * shared/api.js — API client utilities
 */

import { auth } from "../firebase-config.js";

/**
 * Get the API base URL
 * Priority: meta tag api-base > window.SCHOLARAI_API_BASE > inferred from hostname > localhost default
 * @returns {string} API base URL
 */
export function getApiBase() {
  // 1. Check meta tag (injected at deploy time)
  const metaTag = document.querySelector('meta[name="api-base"]');
  if (metaTag && metaTag.content) {
    return metaTag.content;
  }
  
  // 2. Check global variable (set by inline script)
  if (window.SCHOLARAI_API_BASE) {
    return window.SCHOLARAI_API_BASE;
  }
  
  // 3. Local development
  const hostname = window.location.hostname;
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }
  
  // 4. Production: infer from current origin (GitHub Pages -> same domain won't work for API)
  // This is a fallback; meta tag should always be set in production
  return `https://${hostname}`;
}

/**
 * Get the current user's ID token
 * @returns {Promise<string|null>} ID token or null if not authenticated
 */
async function getIdToken() {
  if (!auth || !auth.currentUser) {
    return null;
  }
  try {
    return await auth.currentUser.getIdToken(true); // force refresh
  } catch (e) {
    console.warn("Failed to get ID token:", e);
    return null;
  }
}

/**
 * Make an API request with error handling and auth
 * @param {string} endpoint - API endpoint (e.g., "/generate/summary")
 * @param {Object} options - Fetch options
 * @returns {Promise<Object>} Response data
 * @throws {Error} If request fails
 */
export async function apiRequest(endpoint, options = {}) {
  const base = getApiBase();
  const url = `${base}${endpoint}`;
  
  // Get ID token for authentication
  const idToken = await getIdToken();
  
  const headers = {
    "Content-Type": "application/json",
    ...options.headers
  };
  
  if (idToken) {
    headers["Authorization"] = `Bearer ${idToken}`;
  }
  
  const defaultOptions = {
    headers,
    ...options
  };
  
  const resp = await fetch(url, defaultOptions);
  
  if (!resp.ok) {
    const errorText = await resp.text().catch(() => "");
    throw new Error(`${endpoint} failed: ${resp.status} ${errorText}`);
  }
  
  return resp.json();
}

/**
 * POST request to generate endpoint
 * @param {string} type - Generation type: "summary", "quiz", "audio", "video"
 * @param {string} fileId - File ID from upload
 * @returns {Promise<Object>} Generation result
 */
export async function generateOutput(type, fileId) {
  return apiRequest(`/generate/${type}`, {
    method: "POST",
    body: JSON.stringify({ file_id: fileId })
  });
}

/**
 * Upload file to backend
 * @param {File} file - File to upload
 * @returns {Promise<Object>} Upload result with file_id
 */
export async function uploadFile(file) {
  const base = getApiBase();
  const formData = new FormData();
  formData.append("file", file);
  
  const idToken = await getIdToken();
  const headers = {};
  if (idToken) {
    headers["Authorization"] = `Bearer ${idToken}`;
  }
  
  const resp = await fetch(`${base}/upload`, {
    method: "POST",
    headers,
    body: formData
  });
  
  if (!resp.ok) {
    throw new Error(`Upload failed: ${resp.status}`);
  }
  
  return resp.json();
}

/**
 * Delete file and associated outputs
 * @param {string} fileId - File ID to delete
 * @returns {Promise<Object>} Delete result
 */
export async function cleanupFile(fileId) {
  return apiRequest(`/cleanup/${fileId}`, { method: "DELETE" });
}

/**
 * Create async generation job
 * @param {string} type - Generation type: "summary", "quiz", "audio", "video"
 * @param {string} fileId - File ID from upload
 * @returns {Promise<Object>} Job creation result with job_id
 */
export async function createJob(type, fileId) {
  return apiRequest("/jobs", {
    method: "POST",
    body: JSON.stringify({ file_id: fileId, generation_type: type })
  });
}

/**
 * Get job status
 * @param {string} jobId - Job ID
 * @returns {Promise<Object>} Job status
 */
export async function getJobStatus(jobId) {
  return apiRequest(`/jobs/${jobId}`);
}

/**
 * List user's jobs
 * @returns {Promise<Array>} List of jobs
 */
export async function listJobs() {
  return apiRequest("/jobs");
}

/**
 * Connect to job progress WebSocket
 * @param {string} jobId - Job ID
 * @param {Function} onProgress - Callback for progress updates
 * @returns {WebSocket} WebSocket connection
 */
export function connectJobWebSocket(jobId, onProgress) {
  const base = getApiBase().replace("https://", "wss://").replace("http://", "ws://");
  const url = new URL(`/ws/jobs/${jobId}`, base);
  
  // Add token as query parameter
  getIdToken().then(token => {
    if (token) {
      url.searchParams.set("token", token);
    }
  });
  
  const ws = new WebSocket(url.toString());
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (onProgress) onProgress(data);
    } catch (e) {
      console.warn("Failed to parse WebSocket message:", e);
    }
  };
  
  ws.onerror = (error) => {
    console.error("WebSocket error:", error);
  };
  
  return ws;
}