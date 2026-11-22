from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from visionpilot_api.app import create_app
from visionpilot_api.config import Settings

from .conftest import FakeTools, ScriptedPlanner, call


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 50), "red").save(buf, format="PNG")
    return buf.getvalue()


def _events(body: str) -> list[dict]:
    return [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")]


def test_upload_serve_and_chat_stream(settings: Settings) -> None:
    planner = ScriptedPlanner([call("detect_objects", {}), call("final_answer", {"answer": "2 people"})])
    app = create_app(settings, tool_client=FakeTools(), planner=planner)
    with TestClient(app) as client:
        up = client.post("/api/uploads", files={"file": ("shot.png", _png(), "image/png")})
        assert up.status_code == 200, up.text
        media = up.json()
        assert media["kind"] == "image" and (media["width"], media["height"]) == (100, 50)
        assert client.get(media["url"]).status_code == 200

        resp = client.post("/api/chat", json={"media_id": media["id"], "question": "How many people?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _events(resp.text)
    assert [e["type"] for e in events][-2:] == ["answer", "done"]
    assert events[-2]["answer"]["answer"] == "2 people"


def test_rejects_bad_uploads(settings: Settings) -> None:
    with TestClient(create_app(settings, tool_client=FakeTools(), planner=ScriptedPlanner([]))) as client:
        assert (
            client.post("/api/uploads", files={"file": ("a.exe", b"MZ", "application/octet-stream")}).status_code == 400
        )
        assert client.post("/api/uploads", files={"file": ("fake.png", b"nope", "image/png")}).status_code == 400


def test_media_ids_cannot_escape_folders(settings: Settings) -> None:
    with TestClient(create_app(settings, tool_client=FakeTools(), planner=ScriptedPlanner([]))) as client:
        assert client.get("/api/media/upload:../../.env").status_code == 404
        assert client.get("/api/media/sample:../secrets.txt").status_code == 404
        assert client.post("/api/chat", json={"media_id": "C:/Windows/win.ini", "question": "hi"}).status_code == 404


def test_samples_listing(settings: Settings) -> None:
    with TestClient(create_app(settings, tool_client=FakeTools(), planner=ScriptedPlanner([]))) as client:
        samples = client.get("/api/samples").json()
    assert samples[0]["id"] == "sample:retail/store_queue.jpg"
    assert samples[0]["scenario"] == "retail" and samples[0]["title"] == "Store Queue"


def test_missing_gemini_key_streams_a_clear_error(settings: Settings) -> None:
    with TestClient(create_app(settings, tool_client=FakeTools())) as client:
        resp = client.post("/api/chat", json={"media_id": "sample:retail/store_queue.jpg", "question": "hi"})
        events = _events(resp.text)
    assert events[0]["type"] == "error" and "GEMINI_API_KEY" in events[0]["message"]


def test_health(settings: Settings) -> None:
    with TestClient(create_app(settings, tool_client=FakeTools(), planner=ScriptedPlanner([]))) as client:
        health = client.get("/api/health").json()
    assert health["mcp_connected"] is True and "detect_objects" in health["tools"]
    assert health["gemini_key_set"] is False
