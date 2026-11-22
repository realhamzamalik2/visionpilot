/** Typed client for the VisionPilot FastAPI backend (mirrors backend/src/visionpilot_api/schemas.py). */

export const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export type MediaKind = "image" | "video";

export interface MediaInfo {
  id: string;
  kind: MediaKind;
  filename: string;
  url: string;
  width?: number | null;
  height?: number | null;
  size_bytes: number;
}

export interface SampleInfo extends MediaInfo {
  scenario: string;
  title: string;
  suggested_questions: string[];
}

export interface Evidence {
  step: number;
  item_indices: number[];
  label: string;
}

export interface FinalAnswer {
  answer: string;
  evidence: Evidence[];
}

export type TraceEventType = "run_started" | "thinking" | "tool_call" | "tool_result" | "answer" | "error" | "done";

export interface TraceEvent {
  type: TraceEventType;
  run_id: string;
  step?: number;
  call_id?: string;
  tool?: string;
  args?: Record<string, unknown>;
  ok?: boolean;
  summary?: string;
  latency_ms?: number;
  data?: Record<string, unknown>;
  message?: string;
  answer?: FinalAnswer;
  model?: string;
}

export interface Health {
  status: string;
  mcp_connected: boolean;
  tools: string[];
  gemini_key_set: boolean;
  model: string;
}

export interface ChatTurn {
  role: "user" | "assistant";
  text: string;
}

export const mediaSrc = (media: Pick<MediaInfo, "url">) => `${API_BASE}${media.url}`;

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const getHealth = async () => json<Health>(await fetch(`${API_BASE}/api/health`, { cache: "no-store" }));

export const getSamples = async () => json<SampleInfo[]>(await fetch(`${API_BASE}/api/samples`, { cache: "no-store" }));

export async function uploadMedia(file: File): Promise<MediaInfo> {
  const form = new FormData();
  form.append("file", file);
  return json<MediaInfo>(await fetch(`${API_BASE}/api/uploads`, { method: "POST", body: form }));
}

/**
 * POST /api/chat and parse the Server-Sent Events stream. EventSource only supports GET, so this reads the
 * response body directly and splits it on blank lines.
 */
export async function streamChat(
  body: { media_id: string; question: string; history: ChatTurn[] },
  onEvent: (event: TraceEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    await json(response);
    return;
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = chunk
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data) as TraceEvent);
      boundary = buffer.indexOf("\n\n");
    }
  }
}
