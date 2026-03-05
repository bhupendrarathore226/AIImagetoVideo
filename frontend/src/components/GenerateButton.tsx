import { Loader2, Sparkles } from "lucide-react";

interface Props {
  onClick: () => void;
  disabled: boolean;
  isGenerating: boolean;
}

export default function GenerateButton({
  onClick,
  disabled,
  isGenerating,
}: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="btn-primary w-full py-4 text-base"
    >
      {isGenerating ? (
        <>
          <Loader2 className="w-5 h-5 animate-spin" />
          Generating video…
        </>
      ) : (
        <>
          <Sparkles className="w-5 h-5" />
          Generate Video
        </>
      )}
    </button>
  );
}
