from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageOps

from .config import MAX_IMAGE_PIXELS


def read_image(raw: bytes) -> Image.Image:
    if not raw:
        raise ValueError("Skedari është bosh.")

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError("Skedari nuk është imazh i vlefshëm JPG/PNG.") from exc

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("Imazhi është tepër i madh për përpunim të sigurt.")
    return image


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
