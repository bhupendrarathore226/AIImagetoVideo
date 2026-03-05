import { Loader2, Wand2 } from "lucide-react";

/**
 * Shown while the backend is processing the video generation request.
 * Displays an animated spinner with descriptive pipeline steps.
 */
export default function LoadingIndicator() {
  const steps = [
    "Analyzing image composition",
    "Preparing AI model input",
    "Computing motion vectors",
    "Generating video frames",
    "Encoding final MP4",
  ];

  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-6 py-10 animate-fade-in">
      {/* Animated icon */}
      <div className="relative flex items-center justify-center">
        <div className="w-16 h-16 rounded-full bg-violet-600/20 flex items-center justify-center">
          <Wand2 className="w-7 h-7 text-violet-400" />
        </div>
        <Loader2 className="w-24 h-24 text-violet-600/30 animate-spin absolute" />
      </div>

      {/* Status text */}
      <div className="text-center space-y-1">
        <p className="text-sm font-semibold text-slate-300">
          Generating your video…
        </p>
        <p className="text-xs text-slate-500">
          This typically takes 30–90 seconds
        </p>
      </div>

      {/* Pipeline steps */}
      <div className="w-full max-w-[220px] space-y-2">
        {steps.map((step) => (
          <div key={step} className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-violet-600/50 flex-shrink-0 animate-pulse" />
            <span className="text-xs text-slate-600">{step}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
