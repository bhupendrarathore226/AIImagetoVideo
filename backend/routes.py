"""
API Routes
==========
Defines the HTTP endpoints for the image-to-video pipeline.
"""

import logging
import os
import shutil
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from video_generator import VideoGenerator

logger = logging.getLogger(__name__)

router = APIRouter()

# VideoGenerator is lightweight to instantiate — safe at module level
generator = VideoGenerator()

# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10


@router.get("/api/health")
async def health_check():
    """Returns server health and current AI backend configuration."""
    return {
        "status": "ok",
        "backend": generator.backend,
        "replicate_configured": bool(generator.replicate_token),
    }


@router.post("/api/generate")
async def generate_video(
    image: UploadFile = File(..., description="Source image to animate (JPEG/PNG/WEBP)"),
    prompt: str = Form(default="", description="Motion description, e.g. 'camera slowly zooms in'"),
):
    """
    Generate a short video from an uploaded image.

    - Validates the uploaded file type and size
    - Forwards to the configured AI backend (Replicate or local)
    - Returns a URL to the generated MP4 video
    """
    # ── Validate MIME type ───────────────────────────────────────────────────
    content_type = (image.content_type or "").lower()
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Please upload a JPEG, PNG, or WEBP image.",
        )

    # ── Save upload to temp directory ────────────────────────────────────────
    image_id = str(uuid.uuid4())
    extension = os.path.splitext(image.filename or "upload.jpg")[1] or ".jpg"
    image_path = os.path.join("temp", f"upload_{image_id}{extension}")

    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Validate file size after save
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({file_size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.",
            )

        logger.info(f"Generating video | prompt='{prompt}' | size={file_size_mb:.2f} MB")

        # ── Run AI pipeline ──────────────────────────────────────────────────
        video_path = await generator.generate(image_path, prompt)
        video_filename = os.path.basename(video_path)

        logger.info(f"Video ready: /temp/{video_filename}")

        return JSONResponse({
            "success": True,
            "video_url": f"/temp/{video_filename}",
            "prompt": prompt,
        })

    except HTTPException:
        raise  # Re-raise HTTP errors as-is
    except Exception as exc:
        logger.error(f"Video generation failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Always clean up the uploaded source image
        if os.path.exists(image_path):
            os.remove(image_path)

