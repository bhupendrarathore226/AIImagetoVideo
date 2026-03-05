"""
Video Generator
===============
Core AI pipeline for image-to-video generation.

Supports three backends, configured via the VIDEO_BACKEND environment variable:

┌──────────────────────┬──────────────────────────────────────┬───────────────────────────┐
│ VIDEO_BACKEND value  │ Model                                │ Requirements              │
├──────────────────────┼──────────────────────────────────────┼───────────────────────────┤
│ replicate_svd        │ Stable Video Diffusion (Replicate)   │ REPLICATE_API_TOKEN       │
│ replicate_minimax    │ Minimax Video-01 (Replicate)         │ REPLICATE_API_TOKEN       │
│ huggingface_local    │ SVD XT (local GPU, HuggingFace)      │ ~16 GB VRAM + diffusers   │
└──────────────────────┴──────────────────────────────────────┴───────────────────────────┘

SVD (Stable Video Diffusion) generates 25 frames of smooth, natural motion from a
still image. The text prompt is parsed for motion-intensity keywords and mapped to
SVD's `motion_bucket_id` parameter (1–255).

Minimax Video-01 supports full text-guided animation — the prompt directly controls
the generated motion.
"""

import asyncio
import logging
import os
import uuid

from PIL import Image

logger = logging.getLogger(__name__)


class VideoGenerator:
    def __init__(self):
        self.backend = os.getenv("VIDEO_BACKEND", "replicate_svd").lower()
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN", "")
        self.hf_token = os.getenv("HF_TOKEN", "")

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
        else:
            raise ValueError(
                f"Unknown VIDEO_BACKEND='{self.backend}'. "
                "Choose: replicate_svd | replicate_minimax | huggingface_local"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Replicate backend
    # ──────────────────────────────────────────────────────────────────────────

    async def _generate_replicate(self, image_path: str, prompt: str) -> str:
        """
        Submit a generation request to the Replicate cloud API.

        The image is pre-processed (resized / cropped) to SVD's expected
        1024×576 resolution before upload.
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
                output = await asyncio.to_thread(
                    self._run_svd_replicate, processed_path, prompt
                )
            else:  # replicate_minimax
                output = await asyncio.to_thread(
                    self._run_minimax_replicate, processed_path, prompt
                )

            # Replicate returns a FileOutput object; get its URL
            video_url = output.url if hasattr(output, "url") else str(output)
            await self._download_video(video_url, output_path)

        finally:
            if processed_path != image_path and os.path.exists(processed_path):
                os.remove(processed_path)

        return output_path

    def _run_svd_replicate(self, image_path: str, prompt: str):
        """
        Call stability-ai/stable-video-diffusion on Replicate.

        SVD generates 25 frames of smooth animation from a single image.
        It does not understand text prompts directly — instead we map
        motion-intensity keywords to `motion_bucket_id` (1–255).
        """
        import replicate

        motion_bucket = self._prompt_to_motion_bucket(prompt)
        logger.info(f"SVD — motion_bucket_id={motion_bucket}")

        output = replicate.run(
            "stability-ai/stable-video-diffusion",
            input={
                "input_image": open(image_path, "rb"),
                # Controls camera + object motion (1 = still, 255 = very dynamic)
                "motion_bucket_id": motion_bucket,
                # Conditioning augmentation — small values give cleaner output
                "cond_aug": 0.02,
                "sizing_strategy": "maintain_aspect_ratio",
                "frames_per_second": 6,
                "video_length": "25_frames_with_svd_xt",
                "decoding_t": 14,
            },
        )
        return output

    def _run_minimax_replicate(self, image_path: str, prompt: str):
        """
        Call minimax/video-01 on Replicate.

        Minimax Video-01 is a full multimodal model that accepts both an image
        (first frame anchor) and a text prompt for guided animation.
        """
        import replicate

        effective_prompt = prompt.strip() or "smooth natural camera movement, cinematic"
        logger.info(f"Minimax Video-01 — prompt='{effective_prompt}'")

        output = replicate.run(
            "minimax/video-01",
            input={
                "prompt": effective_prompt,
                "first_frame_image": open(image_path, "rb"),
            },
        )
        return output

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
        import httpx

        logger.info(f"Downloading video from Replicate…")
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
        logger.info(f"Video saved → {output_path}")
