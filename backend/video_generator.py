"""
Video Generator
===============
Core AI pipeline for image-to-video generation.

Supports three backends, configured via the VIDEO_BACKEND environment variable:

┌──────────────────────┬──────────────────────────────────────┬───────────────────────────┐
│ VIDEO_BACKEND value  │ Model                                │ Requirements              │
├──────────────────────┼──────────────────────────────────────┼───────────────────────────┤
│ replicate_minimax    │ Minimax Video-01 (Replicate)         │ REPLICATE_API_TOKEN       │
│ replicate_svd        │ WAN 2.2 Image-to-Video Fast (Replicate) │ REPLICATE_API_TOKEN    │
│ huggingface_local    │ SVD XT (local GPU, HuggingFace)      │ ~16 GB VRAM + diffusers   │
│ comfyui              │ WAN 2.2 i2v (local ComfyUI server)   │ ComfyUI + WAN 2.2 models  │
└──────────────────────┴──────────────────────────────────────┴───────────────────────────┘

The Replicate backends use the REST API directly via httpx — no third-party
replicate package required. This ensures Python 3.14 compatibility.

SVD (Stable Video Diffusion) generates 25 frames of smooth, natural motion from a
still image. The text prompt is parsed for motion-intensity keywords and mapped to
SVD's `motion_bucket_id` parameter (1–255).

Minimax Video-01 supports full text-guided animation — the prompt directly controls
the generated motion.
"""

import asyncio
import base64
import logging
import os
import uuid

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

# Replicate REST API base URL
REPLICATE_API_BASE = "https://api.replicate.com/v1"

# Seconds to wait between prediction status polls
POLL_INTERVAL = 3


class VideoGenerator:
    def __init__(self):
        self.backend = os.getenv("VIDEO_BACKEND", "replicate_svd").lower()
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN", "")
        self.hf_token = os.getenv("HF_TOKEN", "")

        # ComfyUI settings (only used when backend == "comfyui")
        self.comfyui_url = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
        self.comfyui_wan_model = os.getenv("COMFYUI_WAN_MODEL", "wan2.2_i2v_480p_14B_bf16.safetensors")
        self.comfyui_wan_clip = os.getenv("COMFYUI_WAN_CLIP", "umt5-xxl-enc-bf16.safetensors")
        self.comfyui_wan_vae = os.getenv("COMFYUI_WAN_VAE", "wan_2.2_vae.safetensors")

        # Local pipeline is loaded lazily on first use (only for huggingface_local)
        self._local_pipeline = None

        os.makedirs("temp", exist_ok=True)
        logger.info(f"VideoGenerator ready — backend: {self.backend}")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def generate(self, image_path: str, prompt: str) -> str:
        """
        Generate a short video from *image_path* using the configured backend.

        Args:
            image_path: Absolute or relative path to the source image.
            prompt:     Free-form motion description from the user.

        Returns:
            Path to the generated MP4 file (inside the temp/ directory).
        """
        if self.backend in ("replicate_svd", "replicate_minimax"):
            return await self._generate_replicate(image_path, prompt)
        elif self.backend == "huggingface_local":
            return await self._generate_local(image_path, prompt)
        elif self.backend == "comfyui":
            return await self._generate_comfyui(image_path, prompt)
        else:
            raise ValueError(
                f"Unknown VIDEO_BACKEND='{self.backend}'. "
                "Choose: replicate_svd | replicate_minimax | huggingface_local | comfyui"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Replicate backend
    # ──────────────────────────────────────────────────────────────────────────

    async def _generate_replicate(self, image_path: str, prompt: str) -> str:
        """
        Submit a generation request to the Replicate cloud REST API.

        The image is:
        1. Pre-processed (resized / cropped) to SVD's expected resolution.
        2. Base64-encoded and sent as a data URI — no separate upload step.
        """
        if not self.replicate_token:
            raise RuntimeError(
                "REPLICATE_API_TOKEN is not set. "
                "Get a free token at https://replicate.com/account/api-tokens "
                "and add it to backend/.env"
            )

        # Resize happens in a thread to avoid blocking the event loop
        processed_path = await asyncio.to_thread(self._resize_image, image_path)
        output_path = f"temp/video_{uuid.uuid4()}.mp4"

        try:
            if self.backend == "replicate_svd":
                video_url = await self._run_svd_replicate(processed_path, prompt)
            else:  # replicate_minimax
                video_url = await self._run_minimax_replicate(processed_path, prompt)

            await self._download_video(video_url, output_path)

        finally:
            if processed_path != image_path and os.path.exists(processed_path):
                os.remove(processed_path)

        return output_path

    async def _run_svd_replicate(self, image_path: str, prompt: str) -> str:
        """
        Call wan-video/wan-2.2-i2v-fast on Replicate via REST API.

        WAN 2.2 i2v-fast is a PrunaAI-optimised version of Wan 2.2 A14B
        image-to-video — faster and cheaper than the full model.

        Returns the URL of the generated video.
        """
        effective_prompt = prompt.strip() or "smooth natural motion, cinematic"
        logger.info(f"WAN 2.2 i2v-fast — prompt='{effective_prompt}'")

        data_uri = await asyncio.to_thread(self._encode_image_to_data_uri, image_path)

        prediction = await self._create_replicate_prediction(
            model="wan-video/wan-2.2-i2v-fast",
            input_payload={
                "image": data_uri,
                "prompt": effective_prompt,
                "num_frames": 81,       # ~5 seconds at 16 fps
                "go_fast": True,
                "resolution": "480p",
            },
        )

        output = prediction["output"]
        if isinstance(output, list):
            return str(output[0])
        return str(output)

    async def _run_minimax_replicate(self, image_path: str, prompt: str) -> str:
        """
        Call minimax/video-01 on Replicate via REST API.

        Minimax Video-01 is a full multimodal model that accepts both an image
        (first frame anchor) and a text prompt for guided animation.

        Returns the URL of the generated video.
        """
        effective_prompt = prompt.strip() or "smooth natural camera movement, cinematic"
        logger.info(f"Minimax Video-01 — prompt='{effective_prompt}'")

        data_uri = await asyncio.to_thread(self._encode_image_to_data_uri, image_path)

        prediction = await self._create_replicate_prediction(
            model="minimax/video-01",
            input_payload={
                "prompt": effective_prompt,
                "first_frame_image": data_uri,
            },
        )

        output = prediction["output"]
        if isinstance(output, list):
            return str(output[0])
        return str(output)

    async def _create_replicate_prediction(self, model: str, input_payload: dict) -> dict:
        """
        Create a Replicate prediction and poll until it succeeds or fails.

        Strategy:
        1. Try POST /v1/models/{owner}/{name}/predictions  (latest-deployment endpoint)
        2. If that returns 404, the model has no active deployment — fall back to:
             a. GET  /v1/models/{owner}/{name}/versions  → fetch latest version hash
             b. POST /v1/predictions  with {"version": <hash>, "input": ...}

        This handles both Replicate "official deployment" models and
        community/legacy models that only expose versioned endpoints.
        """
        auth_headers = {"Authorization": f"Token {self.replicate_token}"}
        post_headers = {
            **auth_headers,
            "Content-Type": "application/json",
            "Prefer": "wait=60",  # Ask Replicate to wait up to 60s before responding
        }

        async with httpx.AsyncClient(timeout=300.0) as client:

            # ── Step 1: try the model-deployment endpoint ────────────────────
            model_url = f"{REPLICATE_API_BASE}/models/{model}/predictions"
            logger.info(f"Attempting model-deployment endpoint: {model_url}")

            response = await client.post(
                model_url,
                headers=post_headers,
                json={"input": input_payload},
            )

            # ── Step 2: 404 → fall back to versioned endpoint ────────────────
            if response.status_code == 404:
                logger.info(
                    f"Model deployment not found (404). "
                    f"Fetching latest version hash for {model}…"
                )
                version_id = await self._get_latest_model_version(client, model)
                logger.info(f"Using version: {version_id}")

                response = await client.post(
                    f"{REPLICATE_API_BASE}/predictions",
                    headers=post_headers,
                    json={"version": version_id, "input": input_payload},
                )

            # 200/201 = prediction complete or created synchronously
            # 202 = prediction accepted but Prefer: wait timeout elapsed — poll below
            if response.status_code not in (200, 201, 202):
                raise RuntimeError(
                    f"Replicate API error {response.status_code}: {response.text}"
                )

            prediction = response.json()
            logger.info(
                f"Prediction created — id={prediction.get('id')} "
                f"status={prediction.get('status')}"
            )

            # ── Step 3: poll until terminal status ───────────────────────────
            while prediction.get("status") not in ("succeeded", "failed", "canceled"):
                await asyncio.sleep(POLL_INTERVAL)

                poll_response = await client.get(
                    f"{REPLICATE_API_BASE}/predictions/{prediction['id']}",
                    headers=auth_headers,
                )
                poll_response.raise_for_status()
                prediction = poll_response.json()
                logger.info(f"Prediction status: {prediction.get('status')}")

            if prediction["status"] != "succeeded":
                error_msg = prediction.get("error") or "Unknown error from Replicate"
                raise RuntimeError(f"Replicate prediction failed: {error_msg}")

            logger.info("Prediction succeeded")
            return prediction

    async def _get_latest_model_version(
        self, client: httpx.AsyncClient, model: str
    ) -> str:
        """
        Fetch the latest version hash for a model from the Replicate API.

        GET /v1/models/{owner}/{name}/versions returns results sorted newest-first.
        """
        url = f"{REPLICATE_API_BASE}/models/{model}/versions"
        response = await client.get(
            url,
            headers={"Authorization": f"Token {self.replicate_token}"},
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Could not fetch versions for '{model}' "
                f"({response.status_code}): {response.text}"
            )

        data = response.json()
        versions = data.get("results", [])

        if not versions:
            raise RuntimeError(
                f"No versions found for model '{model}'. "
                "The model may have been removed from Replicate."
            )

        latest = versions[0]["id"]
        logger.info(f"Latest version for {model}: {latest}")
        return latest

    # ──────────────────────────────────────────────────────────────────────────
    # Local HuggingFace backend
    # ──────────────────────────────────────────────────────────────────────────

    async def _generate_local(self, image_path: str, prompt: str) -> str:
        """
        Run Stable Video Diffusion locally using HuggingFace Diffusers.

        Requirements:
        - NVIDIA GPU with ≥16 GB VRAM
        - pip install torch diffusers transformers accelerate
        - On first run, ~10 GB model weights are downloaded from HuggingFace
        """
        output_path = await asyncio.to_thread(self._run_svd_local, image_path, prompt)
        return output_path

    def _run_svd_local(self, image_path: str, prompt: str) -> str:
        """
        Local SVD XT inference.  The pipeline is cached after the first call
        so subsequent generations are much faster.
        """
        import torch
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import export_to_video, load_image

        if self._local_pipeline is None:
            logger.info("Loading SVD XT pipeline — first run may take several minutes…")

            self._local_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float16,
                variant="fp16",
                token=self.hf_token or None,
            )
            # CPU offloading lets the model run on GPUs with limited VRAM
            self._local_pipeline.enable_model_cpu_offload()
            self._local_pipeline.unet.enable_forward_chunking()
            logger.info("SVD XT pipeline loaded successfully")

        pipe = self._local_pipeline
        image = load_image(image_path).resize((1024, 576))
        motion_bucket = self._prompt_to_motion_bucket(prompt)

        logger.info(f"Local SVD XT inference — motion_bucket_id={motion_bucket}")

        # Generate 25 frames → ~3.5 seconds at 7 fps
        frames = pipe(
            image,
            num_frames=25,
            decode_chunk_size=8,
            motion_bucket_id=motion_bucket,
            noise_aug_strength=0.1,
            generator=torch.manual_seed(42),
        ).frames[0]

        output_path = f"temp/video_{uuid.uuid4()}.mp4"
        export_to_video(frames, output_path, fps=7)

        logger.info(f"Local video exported → {output_path}")
        return output_path

    # ──────────────────────────────────────────────────────────────────────────
    # ComfyUI backend
    # ──────────────────────────────────────────────────────────────────────────

    async def _generate_comfyui(self, image_path: str, prompt: str) -> str:
        """
        Generate video via a locally-running ComfyUI server.

        Flow:
          1. Upload source image  →  POST /upload/image
          2. Submit WAN 2.2 i2v workflow  →  POST /prompt
          3. Poll until complete  →  GET /history/{prompt_id}
          4. Stream-download output  →  GET /view
        """
        effective_prompt = prompt.strip() or "smooth natural motion, cinematic"
        logger.info(f"ComfyUI WAN 2.2 i2v — prompt='{effective_prompt}'")

        async with httpx.AsyncClient(timeout=600.0) as client:
            # Step 1 — upload image
            uploaded_name = await self._comfyui_upload_image(client, image_path)
            logger.info(f"ComfyUI image uploaded: {uploaded_name}")

            # Step 2 — submit workflow
            workflow = self._build_wan_i2v_workflow(uploaded_name, effective_prompt)
            client_id = str(uuid.uuid4())
            resp = await client.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"ComfyUI /prompt error {resp.status_code}: {resp.text}"
                )
            prompt_id = resp.json()["prompt_id"]
            logger.info(f"ComfyUI prompt queued — id={prompt_id}")

            # Step 3 — poll history
            output_info = await self._comfyui_poll_history(client, prompt_id)

            # Step 4 — download video
            output_path = f"temp/video_{uuid.uuid4()}.mp4"
            view_url = (
                f"{self.comfyui_url}/view"
                f"?filename={output_info['filename']}"
                f"&subfolder={output_info.get('subfolder', '')}"
                f"&type=output"
            )
            await self._download_video(view_url, output_path)

        return output_path

    async def _comfyui_upload_image(
        self, client: httpx.AsyncClient, image_path: str
    ) -> str:
        """Upload an image to ComfyUI and return the filename it was stored as."""
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/png")}
            resp = await client.post(
                f"{self.comfyui_url}/upload/image",
                files=files,
                data={"overwrite": "true"},
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"ComfyUI image upload failed {resp.status_code}: {resp.text}"
            )
        return resp.json()["name"]

    async def _comfyui_poll_history(
        self, client: httpx.AsyncClient, prompt_id: str
    ) -> dict:
        """
        Poll GET /history/{prompt_id} until the job is complete.
        Returns the first video output info dict with keys: filename, subfolder.
        """
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            resp = await client.get(f"{self.comfyui_url}/history/{prompt_id}")
            resp.raise_for_status()
            data = resp.json()

            if prompt_id not in data:
                continue  # not finished yet

            job = data[prompt_id]
            status = job.get("status", {})

            if status.get("status_str") == "error" or status.get("completed") is False:
                messages = status.get("messages", [])
                raise RuntimeError(f"ComfyUI job failed: {messages}")

            # Walk all node outputs to find the first video file
            for node_output in job.get("outputs", {}).values():
                for vid in node_output.get("videos", []):
                    logger.info(f"ComfyUI output video: {vid}")
                    return vid
                for gif in node_output.get("gifs", []):
                    logger.info(f"ComfyUI output gif: {gif}")
                    return gif

            # outputs present but no video yet — keep polling
            if status.get("completed"):
                raise RuntimeError(
                    "ComfyUI job completed but produced no video output. "
                    "Check that VHS_VideoCombine (ComfyUI-VideoHelperSuite) is installed."
                )

    def _build_wan_i2v_workflow(self, image_name: str, prompt: str) -> dict:
        """
        Build a WAN 2.2 image-to-video ComfyUI workflow dict.

        Node layout:
          1  UNETLoader          — WAN 2.2 i2v unet
          2  CLIPLoader          — UMT5 text encoder
          3  VAELoader           — WAN 2.2 VAE
          4  CLIPTextEncode      — positive prompt
          5  CLIPTextEncode      — negative prompt
          6  LoadImage           — source image
          7  WanImageToVideoLatent — i2v latent conditioning
          8  KSampler            — diffusion sampling
          9  VAEDecode           — decode latents → frames
          10 VHS_VideoCombine    — export MP4
        """
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": self.comfyui_wan_model,
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": self.comfyui_wan_clip,
                    "type": "wan",
                    "device": "default",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": self.comfyui_wan_vae},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["2", 0]},
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "worst quality, low quality, blurry, distorted",
                    "clip": ["2", 0],
                },
            },
            "6": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "7": {
                "class_type": "WanImageToVideoLatent",
                "inputs": {
                    "width": 832,
                    "height": 480,
                    "num_frames": 81,
                    "batch_size": 1,
                    "image": ["6", 0],
                    "vae": ["3", 0],
                },
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 0,
                    "steps": 20,
                    "cfg": 6.0,
                    "sampler_name": "dpmpp_2m_sde",
                    "scheduler": "linear_quadratic",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["7", 0],
                },
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
            },
            "10": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": 16,
                    "loop_count": 0,
                    "filename_prefix": "wan_i2v",
                    "format": "video/h264-mp4",
                    "pingpong": False,
                    "save_output": True,
                    "images": ["9", 0],
                },
            },
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _prompt_to_motion_bucket(self, prompt: str) -> int:
        """
        Convert a free-form motion prompt into an SVD motion_bucket_id (1–255).

        Buckets:
          High  (160–200) — fast, dynamic, dramatic motion
          Mid   (100–140) — natural camera moves (default)
          Low   (40–80)   — subtle, gentle, slow motion
        """
        if not prompt:
            return 127  # Moderate motion

        tokens = set(prompt.lower().split())

        HIGH_MOTION = {
            "fast", "quick", "rapid", "dynamic", "energetic",
            "shake", "turbulent", "intense", "dramatic", "action",
            "spin", "rotate", "rush", "burst", "explosive",
        }
        LOW_MOTION = {
            "slow", "gentle", "subtle", "soft", "calm", "still",
            "peaceful", "quiet", "minimal", "slight", "barely",
            "delicate", "smooth", "steady", "stable", "slowly",
        }

        if tokens & HIGH_MOTION:
            return 180
        if tokens & LOW_MOTION:
            return 60
        return 127

    def _encode_image_to_data_uri(self, image_path: str) -> str:
        """
        Read the image file and return a base64-encoded data URI.

        This lets us send the image directly to the Replicate API without
        needing a publicly accessible URL or a separate file-upload step.
        """
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/png")

        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded}"

    def _resize_image(self, image_path: str) -> str:
        """
        Resize and center-crop the image to SVD's expected dimensions.

        Orientation is chosen based on the input aspect ratio:
        - Landscape → 1024 × 576
        - Portrait  →  576 × 1024
        """
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Pick target dimensions that best match the source orientation
        target_w, target_h = (1024, 576) if w >= h else (576, 1024)
        target_ratio = target_w / target_h
        orig_ratio = w / h

        # Center-crop to match target aspect ratio, then resize
        if orig_ratio > target_ratio:
            crop_w = int(h * target_ratio)
            left = (w - crop_w) // 2
            img = img.crop((left, 0, left + crop_w, h))
        elif orig_ratio < target_ratio:
            crop_h = int(w / target_ratio)
            top = (h - crop_h) // 2
            img = img.crop((0, top, w, top + crop_h))

        img = img.resize((target_w, target_h), Image.LANCZOS)

        resized_path = f"temp/resized_{uuid.uuid4()}.png"
        img.save(resized_path, "PNG")
        return resized_path

    async def _download_video(self, url: str, output_path: str) -> None:
        """Stream-download a video from a URL to a local file."""
        logger.info("Downloading video from Replicate…")
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
        logger.info(f"Video saved → {output_path}")
