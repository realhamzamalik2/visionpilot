# VisionPilot agent backend

FastAPI service that answers questions about an image or video. Gemini (free tier) plans which vision tools to
call, the VisionPilot MCP server runs them, and every step streams to the browser as Server-Sent Events.

```
POST /api/uploads        multipart file -> MediaInfo (id, kind, url, size)
GET  /api/samples        bundled demo media with suggested questions
GET  /api/media/{id}     serve an upload or sample
POST /api/chat           {media_id, question, history} -> text/event-stream of TraceEvents
GET  /api/health         MCP connection, tool list, Gemini key present
```

Agent design:
- **Function calling with `mode=ANY`**: every model turn is a tool call or `final_answer`, whose arguments are a
  Pydantic schema (answer + evidence pointing at tool steps), so the UI can link answer text to boxes.
- **Step limit**: on the last step only `final_answer` is allowed, so a run always ends.
- **Retries with backoff and a fallback model** for free-tier 429s and transient 5xx errors.
- **Token budget**: the UI gets full tool output; the planner gets a compacted copy (no polygons, depth maps,
  per-frame boxes).
- **Prompt-injection guard**: the planner never chooses file paths; the media path is always overwritten with the
  user's own upload.

```powershell
cd D:\projects\visionpilot\mcp_server; uv run visionpilot-mcp --transport streamable-http   # terminal 1
cd D:\projects\visionpilot\backend;    uv run visionpilot-api                              # terminal 2
```
