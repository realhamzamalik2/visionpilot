# VisionPilot MCP server

Pretrained, CPU-only vision models exposed as [Model Context Protocol](https://modelcontextprotocol.io) tools.
Works with Claude Desktop (stdio) and with the VisionPilot agent backend (Streamable HTTP).

| Tool | Model | What it returns |
|---|---|---|
| `detect_objects` | Ultralytics YOLO26n (ONNX Runtime) | Boxes, confidences, per-label counts (80 COCO classes) |
| `describe_image` | Florence-2 base-ft (230M, MIT) | English caption at three detail levels |
| `read_korean_text` | PP-OCRv6 det + PP-OCRv5 Korean rec (RapidOCR, ONNX) | Text lines with polygons, in reading order |
| `server_status` | none | Loaded models, cache stats |

Design notes:
- **Lazy loading with an LRU cap** (`MAX_LOADED_MODELS`, default 2) keeps RAM use low on an 8 GB laptop.
- **Result cache** keyed by image hash + tool + arguments, so repeated questions about one image are instant.
- **Structured output**: every tool returns a Pydantic model, so clients get a JSON schema and typed results.
- **Latency log**: every call is appended to `logs/tool_metrics.jsonl`.

## Quick start (Windows PowerShell)

```powershell
cd D:\projects\visionpilot\mcp_server
uv sync
uv run visionpilot-setup     # one-time: export YOLO26n to ONNX, download Florence-2 and Korean OCR
uv run visionpilot-check     # calls every tool through a real MCP session and prints latencies
uv run pytest                # fast unit tests with fake models
uv run pytest -m models      # slow tests with the real models
```

Run the server:

```powershell
uv run visionpilot-mcp                                   # stdio, for Claude Desktop
uv run visionpilot-mcp --transport streamable-http       # http://127.0.0.1:8765/mcp
```

## Licenses

Ultralytics YOLO26 is AGPL-3.0, so this server is distributed under AGPL-3.0.
Florence-2 is MIT. PaddleOCR models are Apache-2.0. RapidOCR is Apache-2.0.
