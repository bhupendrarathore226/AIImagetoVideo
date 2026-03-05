import { Download, ExternalLink } from "lucide-react";

interface Props {
  videoUrl: string;
}

export default function VideoPreview({ videoUrl }: Props) {
  const handleDownload = () => {
    // Force a download via a temporary anchor element
    const link = document.createElement("a");
    link.href = videoUrl;
    link.download = `generated-video-${Date.now()}.mp4`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Video player */}
      <video
        key={videoUrl} // Re-mount when URL changes so autoPlay fires again
        src={videoUrl}
        controls
        autoPlay
        loop
        muted
        playsInline
        className="w-full rounded-lg bg-black"
        style={{ maxHeight: "280px" }}
      >
        Your browser does not support video playback.
      </video>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          type="button"
          onClick={handleDownload}
          className="btn-primary flex-1"
        >
          <Download className="w-4 h-4" />
          Download MP4
        </button>
        <a
          href={videoUrl}
          target="_blank"
          rel="noopener noreferrer"
          title="Open in new tab"
          className="btn-secondary px-4"
        >
          <ExternalLink className="w-4 h-4" />
        </a>
      </div>
    </div>
  );
}
