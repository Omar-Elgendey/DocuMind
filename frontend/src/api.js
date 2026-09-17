// Base URL of the FastAPI backend. Override at build time with
// VITE_API_BASE_URL if the API isn't running on localhost:8000.
const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

/**
 * Gets or creates a persistent session ID stored in localStorage
 */
function getSessionId() {
  let sessionId = localStorage.getItem("documind_session_id");
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem("documind_session_id", sessionId);
  }
  return sessionId;
}

/**
 * Helper to build custom fetch headers including X-Session-ID
 */
function getHeaders(customHeaders = {}) {
  return {
    "X-Session-ID": getSessionId(),
    ...customHeaders,
  };
}

/**
 * Reads a fetch Response as JSON and throws a readable Error if the
 * request failed, using the backend's {"detail": "..."} message when
 * present so callers can show it directly in the UI.
 */
async function handleResponse(response) {
  if (response.status === 204) {
    return null;
  }

  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      (body && typeof body === "object" && body.detail) ||
      (typeof body === "string" && body) ||
      `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  return body;
}

/** GET /health */
export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`, {
    headers: getHeaders(),
  });
  return handleResponse(response);
}

/** GET /documents */
export async function listDocuments({ status, limit = 50, offset = 0 } = {}) {
  const params = new URLSearchParams({ limit, offset });
  if (status) params.set("status", status);

  const response = await fetch(`${API_BASE}/documents?${params.toString()}`, {
    headers: getHeaders(),
  });
  return handleResponse(response);
}

/** GET /documents/{id} */
export async function getDocument(documentId) {
  const response = await fetch(`${API_BASE}/documents/${documentId}`, {
    headers: getHeaders(),
  });
  return handleResponse(response);
}

/** POST /documents (multipart/form-data upload) */
export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/documents`, {
    method: "POST",
    headers: getHeaders(),
    body: formData,
  });
  return handleResponse(response);
}

/** DELETE /documents/{id} */
export async function deleteDocument(documentId) {
  const response = await fetch(`${API_BASE}/documents/${documentId}`, {
    method: "DELETE",
    headers: getHeaders(),
  });
  return handleResponse(response);
}

/** POST /documents/{id}/chat */
export async function chatWithDocument(documentId, question, topK = 5) {
  const response = await fetch(`${API_BASE}/documents/${documentId}/chat`, {
    method: "POST",
    headers: getHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ question, top_k: topK }),
  });
  return handleResponse(response);
}