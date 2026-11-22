from __future__ import annotations

import base64
from pathlib import Path

import pytest
from PIL import Image

from visionpilot_mcp.errors import ImageLoadError
from visionpilot_mcp.imaging import load_image


def test_loads_local_file(sample_image: Path) -> None:
    loaded = load_image(str(sample_image))
    assert loaded.image.mode == "RGB"
    assert (loaded.info.width, loaded.info.height) == (320, 160)
    assert loaded.info.source_type == "file"
    assert len(loaded.info.sha1) == 16


def test_strips_quotes_around_path(sample_image: Path) -> None:
    assert load_image(f'"{sample_image}"').info.width == 320


def test_loads_data_uri(sample_image: Path) -> None:
    uri = "data:image/png;base64," + base64.b64encode(sample_image.read_bytes()).decode()
    loaded = load_image(uri)
    assert loaded.info.source_type == "base64"
    assert loaded.info.sha1 == load_image(str(sample_image)).info.sha1


def test_converts_rgba_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(path)
    assert load_image(str(path)).image.mode == "RGB"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("", "Empty image source"),
        ("Z:/definitely/missing.jpg", "File not found"),
        ("data:image/png,notbase64", "Data URI must look like"),
    ],
)
def test_bad_sources_raise_clear_errors(source: str, message: str) -> None:
    with pytest.raises(ImageLoadError, match=message):
        load_image(source)


def test_rejects_non_image_file(tmp_path: Path) -> None:
    fake = tmp_path / "fake.jpg"
    fake.write_text("not an image")
    with pytest.raises(ImageLoadError, match="not a readable image"):
        load_image(str(fake))


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    doc = tmp_path / "notes.txt"
    doc.write_text("hello")
    with pytest.raises(ImageLoadError, match="Unsupported file type"):
        load_image(str(doc))


def test_urls_can_be_disabled() -> None:
    with pytest.raises(ImageLoadError, match="disabled"):
        load_image("https://example.com/a.jpg", allow_urls=False)
