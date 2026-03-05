import { History, Play } from "lucide-react";
import type { VideoHistoryItem } from "../types";

interface Props {
  items: VideoHistoryItem[];
  onSelect: (item: VideoHistoryItem) => void;
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function VideoHistory({ items, onSelect }: Props) {
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-5">
        <History className="w-4 h-4 text-violet-400" />
        <h2 className="section-label mb-0">
          Recent generations ({items.length})
        </h2>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5 gap-3">
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => onSelect(item)}
            className="
              group relative rounded-lg overflow-hidden text-left
              border border-white/5 hover:border-violet-500/40
              transition-all duration-200
            "
          >
            {/* Source image thumbnail */}
            <div className="aspect-video relative overflow-hidden bg-black">
              <img
                src={item.imageUrl}
                alt={item.prompt || "Generated video"}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              />
              {/* Play overlay */}
              <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <Play className="w-8 h-8 text-white" fill="white" />
              </div>
            </div>

            {/* Metadata */}
            <div className="p-2" style={{ backgroundColor: "#1a1a2e" }}>
              <p className="text-xs text-slate-300 truncate">
                {item.prompt || <span className="italic text-slate-500">No prompt</span>}
              </p>
              <p className="text-xs text-slate-600 mt-0.5">
                {formatTime(item.createdAt)}
              </p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
