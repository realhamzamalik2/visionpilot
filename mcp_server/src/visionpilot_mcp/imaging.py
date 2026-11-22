"""Load images from a local path, an http(s) URL or a base64 data URI."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError

from visionpilot_mcp.errors import ImageLoadError
from visionpilot_mcp.schemas import ImageInfo

MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}

# Fail fast on decompression bombs instead of eating all 8 GB of RAM.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

SourceType = Literal["file", "url", "base64"]


@dataclass(frozen=True)
class LoadedImage:
    image: Image.Image  # RGB, EXIF orientation applied
    info: ImageInfo


def load_image(source: str, *, allow_urls: bool = True, timeout: float = 15.0) -> LoadedImage:
    """Resolve `source` to an RGB PIL image plus metadata.

    Accepts a local file path (Windows or POSIX), an http(s) URL, or a `data:image/...;base64,` URI.
    """
    source = source.strip().strip('"').strip("'")
    if not source:
        raise ImageLoadError("Empty image source. Pass a file path, an http(s) URL or a data URI.")

    if source.startswith("data:"):
        return decode_image_bytes(_decode_data_uri(source), source_type="base64")
    if source.startswith(("http://", "https://")):
        if not allow_urls:
            raise ImageLoadError("Remote URLs are disabled on this server (ALLOW_REMOTE_URLS=false).")
        return decode_image_bytes(_download(source, timeout), source_type="url")
    return decode_image_bytes(_read_file(Path(source)), source_type="file")


def decode_image_bytes(data: bytes, *, source_type: SourceType) -> LoadedImage:
    try:
        with Image.open(io.BytesIO(data)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image.load()
    except Image.DecompressionBombError as exc:
        raise ImageLoadError(f"Image is larger than {MAX_PIXELS:,} pixels.") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError("The data is not a readable image (supported: JPEG, PNG, WebP, BMP, TIFF, GIF).") from exc

    digest = hashlib.sha1(data, usedforsecurity=False).hexdigest()[:16]
    info = ImageInfo(width=image.width, height=image.height, sha1=digest, source_type=source_type)
    return LoadedImage(image=image, info=info)


def _read_file(path: Path) -> bytes:
    path = path.expanduser()
    if not path.is_file():
        raise ImageLoadError(f"File not found: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ImageLoadError(f"Unsupported file type '{path.suffix}'. Expected one of: {sorted(IMAGE_SUFFIXES)}")
    if path.stat().st_size > MAX_BYTES:
        raise ImageLoadError(f"File is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    return path.read_bytes()


def _decode_data_uri(uri: str) -> bytes:
    header, _, payload = uri.partition(",")
    if ";base64" not in header or not payload:
        raise ImageLoadError("Data URI must look like data:image/png;base64,<data>.")
    try:
        data = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ImageLoadError("Invalid base64 in data URI.") from exc
    if len(data) > MAX_BYTES:
        raise ImageLoadError(f"Image is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    return data


def _download(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "VisionPilot/0.1"})  # noqa: S310 (http/https only)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (scheme checked above)
            data: bytes = response.read(MAX_BYTES + 1)
    except OSError as exc:
        raise ImageLoadError(f"Could not download image: {exc}") from exc
    if len(data) > MAX_BYTES:
        raise ImageLoadError(f"Remote image is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    return data
