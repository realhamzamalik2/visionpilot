# Evaluation set

`questions.jsonl` holds 25 questions about the bundled samples, each with:

- **expected tools**: `tools_all` (every one must be called) and/or `tools_any` (at least one must be called);
- **answer checks**: `answer_contains_all`, `answer_contains_any` (case-insensitive), `answer_regex`, and
  `answer_number` (some integer in the answer falls inside `[min, max]`).

Run it with all servers up (MCP on :8765, Gemini key in `.env`):

```powershell
cd D:\projects\visionpilot\backend
uv run visionpilot-eval                 # all 25 (about 75 Gemini requests, within the free tier)
uv run visionpilot-eval --limit 5       # quick smoke run
```

Results are written to `evals/results/latest.json` and `evals/results/latest.md` (the table used in the README).
Metrics: **tool selection accuracy** (expected tools were called), **answer accuracy** (all answer checks pass),
**task success** (both), plus mean tool calls and latency per question.
