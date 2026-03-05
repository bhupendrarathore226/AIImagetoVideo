interface Props {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

const EXAMPLE_PROMPTS = [
  "Camera slowly zooms in",
  "Gentle wind moves the trees",
  "Clouds drift slowly across the sky",
  "Camera pans left smoothly",
  "Water ripples gently",
  "Dramatic fast zoom out",
  "Soft bokeh depth-of-field blur",
  "Birds fly through the scene",
];

export default function PromptInput({
  value,
  onChange,
  disabled = false,
}: Props) {
  return (
    <div className="space-y-3">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder='Describe the desired motion, e.g. "camera slowly zooms in while clouds move"…'
        disabled={disabled}
        rows={3}
        className="input-base resize-none"
        maxLength={300}
      />

      {/* Character counter */}
      <div className="flex justify-end">
        <span className="text-xs text-slate-600">{value.length}/300</span>
      </div>

      {/* Example prompt chips */}
      <div>
        <p className="text-xs text-slate-500 mb-2">Try an example:</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLE_PROMPTS.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => !disabled && onChange(example)}
              disabled={disabled}
              className="
                text-xs px-3 py-1.5 rounded-full
                border border-white/10 bg-white/[0.02]
                text-slate-400 hover:text-violet-300 hover:border-violet-500/40
                hover:bg-violet-500/5 transition-all duration-150
                disabled:opacity-40 disabled:cursor-not-allowed
              "
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
