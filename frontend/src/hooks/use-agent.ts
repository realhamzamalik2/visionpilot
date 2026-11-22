"use client";

import { useCallback, useRef, useState } from "react";

import { type ChatTurn, type FinalAnswer, type MediaInfo, streamChat, type TraceEvent } from "@/lib/api";
import type { StepResult } from "@/lib/overlays";

export type StepStatus = "running" | "ok" | "error";

export interface ToolStep {
  callId: string;
  step: number;
  tool: string;
  args: Record<string, unknown>;
  status: StepStatus;
  summary?: string;
  latencyMs?: number;
  data?: Record<string, unknown>;
}

export interface Run {
  id: string;
  mediaId: string;
  question: string;
  status: "running" | "done" | "error";
  thinking?: string;
  steps: ToolStep[];
  answer?: FinalAnswer;
  error?: string;
  model?: string;
  latencyMs?: number;
}

function applyEvent(run: Run, event: TraceEvent): Run {
  switch (event.type) {
    case "run_started":
      return { ...run, model: event.model };
    case "thinking":
      return { ...run, thinking: event.message };
    case "tool_call":
      return {
        ...run,
        thinking: undefined,
        steps: [
          ...run.steps,
          { callId: event.call_id ?? `${event.step}`, step: event.step ?? 0, tool: event.tool ?? "", args: event.args ?? {}, status: "running" },
        ],
      };
    case "tool_result": {
      const id = event.call_id ?? `${event.step}`;
      const existing = run.steps.some((s) => s.callId === id);
      const update = (s: ToolStep): ToolStep => ({
        ...s,
        status: event.ok ? "ok" : "error",
        summary: event.summary,
        latencyMs: event.latency_ms,
        data: event.data,
      });
      const steps = existing
        ? run.steps.map((s) => (s.callId === id ? update(s) : s))
        : [...run.steps, update({ callId: id, step: event.step ?? 0, tool: event.tool ?? "", args: {}, status: "running" })];
      return { ...run, steps };
    }
    case "answer":
      return { ...run, thinking: undefined, answer: event.answer, model: event.model ?? run.model };
    case "error":
      return { ...run, thinking: undefined, error: event.message };
    case "done":
      return { ...run, thinking: undefined, status: run.error && !run.answer ? "error" : "done", latencyMs: event.latency_ms };
  }
}

export function stepResults(run?: Run): StepResult[] {
  if (!run) return [];
  return run.steps.filter((s) => s.status === "ok" && s.data).map((s) => ({ step: s.step, tool: s.tool, data: s.data! }));
}

export function useAgent() {
  const [runs, setRuns] = useState<Run[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const ask = useCallback(
    async (media: MediaInfo, question: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      const localId = crypto.randomUUID();
      const history: ChatTurn[] = runs
        .filter((r) => r.mediaId === media.id && r.answer)
        .slice(-3)
        .flatMap((r) => [
          { role: "user" as const, text: r.question },
          { role: "assistant" as const, text: r.answer!.answer },
        ]);
      setRuns((prev) => [...prev, { id: localId, mediaId: media.id, question, status: "running", steps: [] }]);
      const patch = (fn: (run: Run) => Run) => setRuns((prev) => prev.map((r) => (r.id === localId ? fn(r) : r)));

      try {
        await streamChat({ media_id: media.id, question, history }, (event) => patch((r) => applyEvent(r, event)), controller.signal);
        patch((r) => (r.status === "running" ? { ...r, status: r.answer ? "done" : "error", error: r.error ?? (r.answer ? undefined : "The run ended without an answer.") } : r));
      } catch (err) {
        if (controller.signal.aborted) {
          patch((r) => ({ ...r, status: "error", error: "Stopped." }));
        } else {
          const message = err instanceof Error ? err.message : String(err);
          patch((r) => ({ ...r, status: "error", error: `Could not reach the VisionPilot API (${message}). Is the backend running on port 8000?` }));
        }
      }
    },
    [runs],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);
  const clear = useCallback(() => setRuns([]), []);

  return { runs, ask, stop, clear, busy: runs.some((r) => r.status === "running") };
}
