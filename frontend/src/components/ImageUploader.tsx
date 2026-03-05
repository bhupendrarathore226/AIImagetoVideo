import { useCallback, useRef } from "react";
import { ImageIcon, Upload, X } from "lucide-react";

interface Props {
  onImageSelect: (file: File | null, previewUrl: string) => void;
  preview: string;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
const MAX_SIZE_MB = 10;

export default function ImageUploader({
  onImageSelect,
  preview,
  disabled = false,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    (file: File) => {
      if (!ACCEPTED_TYPES.includes(file.type)) {
        alert("Please upload a JPEG, PNG, or WEBP image.");
        return;
      }
      if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        alert(`Image must be under ${MAX_SIZE_MB} MB.`);
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        onImageSelect(file, e.target?.result as string);
      };
      reader.readAsDataURL(file);
    },
    [onImageSelect]
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    e.target.value = ""; // Reset so the same file can be re-selected
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (disabled) return;
      const file = e.dataTransfer.files?.[0];
      if (file) processFile(file);
    },
    [disabled, processFile]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === "Enter" || e.key === " ") && !disabled) {
      e.preventDefault();
      inputRef.current?.click();
    }
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    onImageSelect(null, "");
  };

  return (
    <div className="relative">
      {preview ? (
        /* ── Preview ──────────────────────────────────────────────────── */
        <div className="relative rounded-lg overflow-hidden group">
          <img
            src={preview}
            alt="Selected source image"
            className="w-full h-52 object-cover rounded-lg"
          />
          {/* Hover overlay — change image */}
          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
            <button
              type="button"
              onClick={() => !disabled && inputRef.current?.click()}
              className="bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg text-sm font-medium backdrop-blur-sm transition-colors"
              disabled={disabled}
            >
              Change image
            </button>
          </div>
          {/* Clear button */}
          {!disabled && (
            <button
              type="button"
              onClick={handleClear}
              title="Remove image"
              className="absolute top-2 right-2 p-1.5 bg-black/60 hover:bg-black/80 text-white rounded-full transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      ) : (
        /* ── Drop zone ─────────────────────────────────────────────────── */
        <div
          role="button"
          tabIndex={disabled ? -1 : 0}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => !disabled && inputRef.current?.click()}
          onKeyDown={handleKeyDown}
          className={`
            border-2 border-dashed rounded-lg h-52
            flex flex-col items-center justify-center gap-3
            transition-all duration-200 cursor-pointer select-none
            ${
              disabled
                ? "border-white/5 opacity-50 cursor-not-allowed"
                : "border-white/10 hover:border-violet-500/50 hover:bg-violet-500/5"
            }
          `}
        >
          <div className="p-3 bg-violet-600/15 rounded-full">
            <Upload className="w-6 h-6 text-violet-400" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-slate-300">
              Drop image here or{" "}
              <span className="text-violet-400 underline underline-offset-2">
                browse
              </span>
            </p>
            <p className="text-xs text-slate-500 mt-1">
              JPEG, PNG, WEBP · Max {MAX_SIZE_MB} MB
            </p>
          </div>
          <div className="flex items-center gap-1.5 text-slate-600 text-xs">
            <ImageIcon className="w-3.5 h-3.5" />
            Landscape images (16:9) work best with SVD
          </div>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        onChange={handleFileChange}
        className="hidden"
        aria-label="Upload image"
        disabled={disabled}
      />
    </div>
  );
}
