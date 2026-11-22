# VisionPilot MCP server

Pretrained, CPU-only vision models exposed as [Model Context Protocol](https://modelcontextprotocol.io) tools.
Works with Claude Desktop (stdio) and with the VisionPilot agent backend (Streamable HTTP).

| Tool | Model | What it returns |
|---|---|---|
| `detect_objects` | YOLO26n (ONNX Runtime) | Boxes, confidences, per-label counts (80 COCO classes) |
| `segment_objects` | YOLO26n-seg (ONNX) | Mask polygons, area covered per label |
| `estimate_pose` | YOLO26n-pose (ONNX) | 17 keypoints per person + rough posture |
| `estimate_depth` | YOLO26n-depth (ONNX) | Metric depth stats, depth inside given boxes, colorized depth map |
| `describe_image` | Florence-2 base-ft (230M, MIT) | English caption at three detail levels |
| `find_phrase` | Florence-2 phrase grounding | Boxes for free-text descriptions |
| `read_text` | Florence-2 OCR | English/Latin text lines with quads |
| `read_korean_text` | PP-OCRv6 det + PP-OCRv5 Korean rec (RapidOCR, ONNX) | Korean text lines in reading order |
| `analyze_faces` | MediaPipe BlazeFace (tiled) + HSEmotion (ONNX) | Faces + apparent expression probabilities |
| `process_video` | OpenCV sampling + YOLO26n or faces | Timeline of counts, peak time, how long the current count lasted |
| `server_status` | none | Loaded models, RAM budget, cache stats |

Design notes:
- **Lazy loading under a RAM budget** (`MODEL_MEMORY_BUDGET_MB`, default 2500): least recently used models are
  evicted first, and models already in memory keep serving while another one loads.
- **Background warm-up** (`WARMUP_MODELS`) so the first question does not wait for Florence-2 to load.
- **Result cache** keyed by image hash + tool + arguments, so repeated questions about one image are instant.
- **Structured output**: every tool returns a Pydantic model, so clients get a JSON schema and typed results.
- **Latency log**: every call is appended to `logs/tool_metrics.jsonl`.

## Quick start (Windows PowerShell)

```powershell
cd D:\projects\visionpilot\mcp_server
uv sync
uv run visionpilot-setup     # one-time: export 4 YOLO26n models to ONNX, download Florence-2, OCR, face models
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
Florence-2 is MIT. PaddleOCR models, RapidOCR, MediaPipe and HSEmotion are Apache-2.0.
