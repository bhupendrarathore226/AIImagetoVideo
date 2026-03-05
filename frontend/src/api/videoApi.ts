import type { GenerateResponse, HealthResponse } from "../types";

const API_BASE = "/api";

/**
 * Submit an image + optional prompt to the backend to generate a video.
 *
 * @param image  - The image file selected by the user
 * @param prompt - Free-form motion description (may be empty)
 * @returns      GenerateResponse containing the relative video URL
 * @throws       Error with a user-readable message on failure
 */
export async function generateVideo(
  image: File,
  prompt: string
): Promise<GenerateResponse> {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("prompt", prompt.trim());

  const response = await fetch(`${API_BASE}/generate`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    // The FastAPI error handler returns { detail: "…" }
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      (errorData as { detail?: string }).detail ??
        `Request failed with status ${response.status}`
    );
  }

  return response.json() as Promise<GenerateResponse>;
}

/**
 * Check whether the backend is reachable and properly configured.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  return response.json() as Promise<HealthResponse>;
}
