import { useCallback, useEffect, useState } from "react";
import { Film } from "lucide-react";

import GenerateButton from "./components/GenerateButton";
import ImageUploader from "./components/ImageUploader";
import LoadingIndicator from "./components/LoadingIndicator";
import PromptInput from "./components/PromptInput";
import VideoHistory from "./components/VideoHistory";
import VideoPreview from "./components/VideoPreview";
import { generateVideo, checkHealth } from "./api/videoApi";
import type { AppStatus, HealthResponse, VideoHistoryItem } from "./types";

export default function App() {
  // ── State ────────────────────────────────────────────────────────────────
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string>("");
  const [prompt, setPrompt] = useState<string>("");
  const [status, setStatus] = useState<AppStatus>("idle");
  const [videoUrl, setVideoUrl] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [history, setHistory] = useState<VideoHistoryItem[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  // ── Check backend health on mount ────────────────────────────────────────
  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleImageSelect = useCallback((file: File | null, previewUrl: string) => {
    setSelectedImage(file);
    setImagePreviewUrl(previewUrl);
    // Reset output state when a new image is chosen
    setVideoUrl("");
    setErrorMessage("");
    setStatus("idle");
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!selectedImage) return;

    setStatus("generating");
    setVideoUrl("");
    setErrorMessage("");

    try {
      const result = await generateVideo(selectedImage, prompt);
      setVideoUrl(result.video_url);
      setStatus("success");

      // Prepend to history (keep last 10 entries)
      setHistory((prev) => [
        {
          id: Date.now().toString(),
          imageUrl: imagePreviewUrl,
          prompt,
          videoUrl: result.video_url,
          createdAt: new Date(),
        },
        ...prev.slice(0, 9),
      ]);
    } catch (err) {
      setStatus("error");
      setErrorMessage(
        err instanceof Error ? err.message : "An unexpected error occurred."
      );
    }
  }, [selectedImage, prompt, imagePreviewUrl]);

  const handleHistorySelect = useCallback((item: VideoHistoryItem) => {
    setVideoUrl(item.videoUrl);
    setPrompt(item.prompt);
    setStatus("success");
  }, []);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen" style={{ backgroundColor: "#0d0d1a" }}>
      {/* ── Header ── */}
      <header
        className="sticky top-0 z-10 border-b border-white/5 backdrop-blur-sm"
        style={{ backgroundColor: "rgba(26,26,46,0.8)" }}
      >
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-violet-600/20 rounded-lg">
              <Film className="w-5 h-5 text-violet-400" />
            </div>
            <div>
              <h1 className="text-base font-bold text-white leading-tight">
                Image → Video
              </h1>
              <p className="text-xs text-slate-500">AI-powered animation POC</p>
            </div>
          </div>

          {/* Backend badge */}
          {health && (
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  health.replicate_configured ? "bg-emerald-400" : "bg-amber-400"
                }`}
              />
              <span className="text-xs text-slate-500">
                {health.backend}
                {!health.replicate_configured && " — no API key"}
              </span>
            </div>
          )}
        </div>
      </header>

      {/* ── Main ── */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* ── Left: Input panel ── */}
          <section className="space-y-5">
            <div className="card">
              <p className="section-label">1 · Upload image</p>
              <ImageUploader
                onImageSelect={handleImageSelect}
                preview={imagePreviewUrl}
                disabled={status === "generating"}
              />
            </div>

            <div className="card">
              <p className="section-label">2 · Describe motion (optional)</p>
              <PromptInput
                value={prompt}
                onChange={setPrompt}
                disabled={status === "generating"}
              />
            </div>

            <GenerateButton
              onClick={handleGenerate}
              disabled={!selectedImage || status === "generating"}
              isGenerating={status === "generating"}
            />

            {status === "error" && errorMessage && (
              <div className="animate-fade-in rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400">
                <span className="font-semibold">Error: </span>
                {errorMessage}
              </div>
            )}
          </section>

          {/* ── Right: Output panel ── */}
          <section>
            <div className="card min-h-[320px] flex flex-col">
              <p className="section-label">3 · Generated video</p>

              {status === "generating" && <LoadingIndicator />}

              {status === "success" && videoUrl && (
                <VideoPreview videoUrl={videoUrl} />
              )}

              {status !== "generating" && !videoUrl && (
                <div className="flex-1 flex flex-col items-center justify-center gap-3 text-slate-600 py-8">
                  <Film className="w-14 h-14 opacity-20" />
                  <p className="text-sm text-center">
                    Upload an image and click{" "}
                    <span className="text-violet-400">Generate</span> to create
                    your video
                  </p>
                </div>
              )}
            </div>
          </section>
        </div>

        {/* ── History ── */}
        {history.length > 0 && (
          <section className="mt-10">
            <VideoHistory items={history} onSelect={handleHistorySelect} />
          </section>
        )}
      </main>
    </div>
  );
}
