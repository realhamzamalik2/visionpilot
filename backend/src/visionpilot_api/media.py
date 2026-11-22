"""Uploads and bundled sample media. Media is referenced by id, never by a client-supplied path."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import BinaryIO

from PIL import Image

from visionpilot_api.schemas import MediaInfo, MediaKind, SampleInfo

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}
_UPLOAD_ID = re.compile(r"^[0-9a-f]{32}\.[a-z0-9]{2,5}$")
_SAMPLE_ID = re.compile(r"^[a-z0-9_-]+/[A-Za-z0-9_.-]+$")


class MediaError(ValueError):
    pass


def kind_for(path: Path) -> MediaKind:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    raise MediaError(f"Unsupported file type '{ext}'. Images: {sorted(IMAGE_EXTS)}, videos: {sorted(VIDEO_EXTS)}")


def _dims(path: Path, kind: MediaKind) -> tuple[int | None, int | None]:
    if kind != "image":
        return None, None
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except OSError as exc:
        raise MediaError("The file is not a readable image.") from exc


class MediaStore:
    def __init__(self, upload_dir: Path, samples_dir: Path, max_upload_mb: int) -> None:
        self.upload_dir = upload_dir
        self.samples_dir = samples_dir
        self.max_bytes = max_upload_mb * 1024 * 1024
        upload_dir.mkdir(parents=True, exist_ok=True)

    # ---- uploads --------------------------------------------------------------------------------
    def save_upload(self, filename: str, stream: BinaryIO) -> MediaInfo:
        ext = Path(filename or "").suffix.lower()
        kind = kind_for(Path(f"x{ext}"))
        name = f"{uuid.uuid4().hex}{ext}"
        dest = self.upload_dir / name
        size = 0
        with dest.open("wb") as out:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if size > self.max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise MediaError(f"File is larger than {self.max_bytes // (1024 * 1024)} MB.")
                out.write(chunk)
        try:
            width, height = _dims(dest, kind)
        except MediaError:
            dest.unlink(missing_ok=True)
            raise
        return MediaInfo(
            id=f"upload:{name}", kind=kind, filename=filename, url=f"/api/media/upload:{name}",
            width=width, height=height, size_bytes=size,
        )  # fmt: skip

    # ---- resolution -----------------------------------------------------------------------------
    def resolve(self, media_id: str) -> tuple[Path, MediaKind]:
        """Map a media id to an absolute path inside our own folders (blocks path traversal)."""
        source, _, rest = media_id.partition(":")
        if source == "upload" and _UPLOAD_ID.match(rest):
            path = self.upload_dir / rest
        elif source == "sample" and _SAMPLE_ID.match(rest):
            path = self.samples_dir / rest
        else:
            raise MediaError(f"Unknown media id '{media_id}'.")
        if not path.is_file():
            raise MediaError(f"Media '{media_id}' not found.")
        return path.resolve(), kind_for(path)

    def info(self, media_id: str) -> MediaInfo:
        path, kind = self.resolve(media_id)
        width, height = _dims(path, kind)
        return MediaInfo(
            id=media_id, kind=kind, filename=path.name, url=f"/api/media/{media_id}",
            width=width, height=height, size_bytes=path.stat().st_size,
        )  # fmt: skip

    # ---- samples --------------------------------------------------------------------------------
    def list_samples(self) -> list[SampleInfo]:
        """Samples live in samples/<scenario>/. An optional samples/manifest.json adds titles and questions."""
        manifest_path = self.samples_dir / "manifest.json"
        manifest: dict[str, dict[str, object]] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples: list[SampleInfo] = []
        if not self.samples_dir.is_dir():
            return samples
        for scenario_dir in sorted(p for p in self.samples_dir.iterdir() if p.is_dir()):
            for file in sorted(scenario_dir.iterdir()):
                if file.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
                    continue
                media_id = f"sample:{scenario_dir.name}/{file.name}"
                meta = manifest.get(f"{scenario_dir.name}/{file.name}", {})
                info = self.info(media_id)
                samples.append(
                    SampleInfo(
                        **info.model_dump(),
                        scenario=scenario_dir.name,
                        title=str(meta.get("title", file.stem.replace("_", " ").title())),
                        suggested_questions=list(meta.get("questions", [])),  # type: ignore[call-overload]
                    )
                )
        return samples
