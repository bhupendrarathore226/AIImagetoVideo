# Image → Video POC

Convert a single still image into a short AI-generated video (3–5 seconds) using
**Stable Video Diffusion** via the Replicate cloud API — no GPU required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Browser  (React + Vite)                   │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │ImageUploader │  │PromptInput  │  │   VideoPreview     │  │
│  └──────┬───────┘  └──────┬──────┘  └────────▲───────────┘  │
│         └────────────┬────┘                  │              │
└──────────────────────┼───────────────────────┼──────────────┘
                       │ POST /api/generate    │ /temp/video.mp4
              multipart│ (image + prompt)      │
┌──────────────────────▼───────────────────────┼──────────────┐
│                FastAPI  Backend               │              │
│  ┌────────────────┐    ┌─────────────────────┴──────────┐   │
│  │   routes.py    │───▶│       video_generator.py        │   │
│  │ (API endpoint) │    │   (AI pipeline orchestration)   │   │
│  └────────────────┘    └──────────────┬─────────────────┘   │
└─────────────────────────────────────┬─┘                     │
                                      │                        │
                           ┌──────────▼──────────┐            │
                           │   Replicate Cloud   │            │
                           │ Stable Video Diff.  │            │
                           │ (or local GPU SVD)  │            │
                           └─────────────────────┘
```

---

## Quick start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| Replicate account | [Free tier](https://replicate.com) |

---

### 1 — Backend setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Copy and edit environment variables
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux

# Open .env and set your REPLICATE_API_TOKEN
```

### 2 — Frontend setup

```bash
cd frontend
npm install
```

### 3 — Run

Open **two terminals**:

**Terminal 1 — Backend**
```bash
cd backend
uvicorn main:app --reload
# API available at http://localhost:8000
# Swagger docs at  http://localhost:8000/docs
```

**Terminal 2 — Frontend**
```bash
cd frontend
npm run dev
# UI available at http://localhost:5173
```

---

## Configuration

Edit `backend/.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `VIDEO_BACKEND` | AI backend to use | `replicate_svd` |
| `REPLICATE_API_TOKEN` | Replicate API key | *(required)* |
| `HF_TOKEN` | HuggingFace token | *(optional)* |

### Backend options

| `VIDEO_BACKEND` | Model | Requirements | Notes |
|-----------------|-------|--------------|-------|
| `replicate_svd` | Stable Video Diffusion | `REPLICATE_API_TOKEN` | Default. No GPU needed. |
| `replicate_minimax` | Minimax Video-01 | `REPLICATE_API_TOKEN` | Full text-prompt support. |
| `huggingface_local` | SVD XT (local) | ≥16 GB VRAM + `diffusers` | No API cost. |

---

## How it works

```
User uploads image
       │
       ▼
Backend validates & saves to temp/
       │
       ▼
Image resized → 1024×576 (SVD optimal resolution)
       │
       ▼
Text prompt parsed for motion keywords
       │  "slowly" → motion_bucket = 60
       │  (default) → motion_bucket = 127
       │  "fast"   → motion_bucket = 180
       ▼
Replicate API called with image + motion_bucket_id
       │  SVD generates 25 frames of smooth animation
       ▼
MP4 downloaded to backend/temp/
       │
       ▼
Frontend receives /temp/video_<id>.mp4
       │
       ▼
Video previewed in browser + available for download
```

---

## Project structure

```
image-to-video-poc/
│
├── backend/
│   ├── main.py              ← FastAPI app + CORS + static file serving
│   ├── routes.py            ← POST /api/generate  |  GET /api/health
│   ├── video_generator.py   ← AI pipeline (Replicate SVD / Minimax / local)
│   ├── requirements.txt
│   ├── .env.example         ← Copy to .env and fill in your API token
│   └── temp/                ← Generated videos (gitignored)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUploader.tsx    ← Drag & drop + file picker
│   │   │   ├── PromptInput.tsx      ← Textarea + example prompt chips
│   │   │   ├── GenerateButton.tsx   ← Submit with loading state
│   │   │   ├── VideoPreview.tsx     ← HTML5 player + download button
│   │   │   ├── VideoHistory.tsx     ← Grid of past generations
│   │   │   └── LoadingIndicator.tsx ← Animated progress display
│   │   ├── api/
│   │   │   └── videoApi.ts          ← fetch wrappers for backend API
│   │   ├── types/
│   │   │   └── index.ts             ← TypeScript interfaces
│   │   ├── App.tsx                  ← Root component + state management
│   │   ├── main.tsx                 ← React entry point
│   │   └── index.css                ← Tailwind + custom CSS components
│   ├── package.json
│   ├── vite.config.ts       ← Dev server proxy → localhost:8000
│   └── tailwind.config.js
│
└── README.md
```

---

## API reference

### `POST /api/generate`

Generate a video from an image.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | ✅ | JPEG / PNG / WEBP, max 10 MB |
| `prompt` | string | ❌ | Motion description |

**Response** — `200 OK`

```json
{
  "success": true,
  "video_url": "/temp/video_abc123.mp4",
  "prompt": "camera slowly zooms in"
}
```

**Example with curl**

```bash
curl -X POST http://localhost:8000/api/generate \
  -F "image=@/path/to/photo.jpg" \
  -F "prompt=camera slowly zooms in while clouds move" \
  | python -m json.tool
```

### `GET /api/health`

```bash
curl http://localhost:8000/api/health
```

```json
{
  "status": "ok",
  "backend": "replicate_svd",
  "replicate_configured": true
}
```

---

## Tech stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 18 + TypeScript | UI components |
| Frontend | Vite | Build tool + dev server with API proxy |
| Frontend | Tailwind CSS | Utility-first styling |
| Frontend | lucide-react | Icon library |
| Backend | FastAPI | Async REST API |
| Backend | Uvicorn | ASGI server |
| Backend | Python-multipart | File upload parsing |
| Backend | httpx | Async HTTP (video download) |
| Backend | Pillow | Image resizing |
| AI | Stable Video Diffusion | Image → video model |
| AI Platform | Replicate | Cloud model inference |

---

## Troubleshooting

**`REPLICATE_API_TOKEN` error**
→ Make sure you copied `.env.example` to `.env` and set your token.

**Port 8000 already in use**
```bash
uvicorn main:app --reload --port 8001
# Also update frontend/vite.config.ts proxy target to :8001
```

**Slow generation (30–90 seconds)**
→ Normal for cloud APIs. Replicate queues requests during high traffic.

**`huggingface_local` — out of memory**
→ The model needs ≥16 GB VRAM. Enable CPU offloading (already in code) or try a smaller batch.

**Installing local backend dependencies**
```bash
# Install PyTorch for your CUDA version first:
# https://pytorch.org/get-started/locally/
pip install torch
pip install diffusers transformers accelerate
```
