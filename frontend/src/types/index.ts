/** Represents a single completed video generation stored in history */
export interface VideoHistoryItem {
  id: string;
  /** Data URL of the source image (for thumbnail display) */
  imageUrl: string;
  /** Motion prompt the user entered */
  prompt: string;
  /** Relative URL of the generated video served by the backend */
  videoUrl: string;
  createdAt: Date;
}

/** Shape of the JSON response from POST /api/generate */
export interface GenerateResponse {
  success: boolean;
  video_url: string;
  prompt: string;
}

/** Shape of GET /api/health response */
export interface HealthResponse {
  status: string;
  backend: string;
  replicate_configured: boolean;
}

/** Application lifecycle states */
export type AppStatus = "idle" | "generating" | "success" | "error";
