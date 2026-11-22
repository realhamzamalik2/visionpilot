"use client";

import { ArrowUp, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Answer } from "@/components/answer";
import { Trace } from "@/components/trace";
import { Button } from "@/components/ui/button";
import type { Run } from "@/hooks/use-agent";
import type { Evidence } from "@/lib/api";
import { cn, formatMs } from "@/lib/utils";

interface Props {
  runs: Run[];
  activeRunId: string | null;
  onSelectRun: (id: string) => void;
  suggestions: string[];
  hasMedia: boolean;
  busy: boolean;
  onAsk: (question: string) => void;
  onStop: () => void;
  onHighlight: (evidence: Evidence | null, runId: string) => void;
  onSelectEvidence: (evidence: Evidence, runId: string) => void;
}

const DEFAULT_SUGGESTIONS = [
  "How many people are here, and how far away is the closest one?",
  "Read any Korean text and translate it to English.",
  "Describe the scene, then count every object you can find.",
];

export function ChatPanel({
  runs,
  activeRunId,
  onSelectRun,
  suggestions,
  hasMedia,
  busy,
  onAsk,
  onStop,
  onHighlight,
  onSelectEvidence,
}: Props) {
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const lastRun = runs.at(-1);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [runs.length, lastRun?.steps.length, lastRun?.answer]);

  const submit = (question: string) => {
    const q = question.trim();
    if (!q || busy || !hasMedia) return;
    onAsk(q);
    setDraft("");
  };

  const chips = suggestions.length ? suggestions : DEFAULT_SUGGESTIONS;

  return (
    <section className="flex h-full min-h-0 flex-col rounded-xl border border-line bg-panel" aria-label="Conversation">
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {runs.length === 0 ? (
          <div className="flex h-full flex-col justify-end gap-3 pb-2">
            <div>
              <h2 className="text-lg font-semibold">Ask about what you see</h2>
              <p className="max-w-prose text-sm text-muted">
                {hasMedia
                  ? "The agent picks which vision models to run, shows each step as it goes, and links its answer to the regions it found."
                  : "Choose a sample or upload your own image or video to start."}
              </p>
            </div>
            {hasMedia && (
              <div className="flex flex-col items-start gap-2">
                {chips.map((s) => (
                  <button
                    key={s}
                    onClick={() => submit(s)}
                    className="rounded-lg border border-line px-3 py-2 text-left text-sm text-muted transition-colors hover:border-focus hover:text-text"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-6">
            {runs.map((run) => (
              <article
                key={run.id}
                className={cn("flex flex-col gap-3", runs.length > 1 && run.id !== activeRunId && "opacity-70")}
                onClick={() => onSelectRun(run.id)}
              >
                <p className="self-end rounded-xl rounded-br-sm bg-panel-2 px-3.5 py-2 text-[15px]">{run.question}</p>
                <Trace run={run} />
                {run.answer && (
                  <div className="flex flex-col gap-1.5">
                    <Answer
                      answer={run.answer}
                      onHighlight={(ev) => onHighlight(ev, run.id)}
                      onSelect={(ev) => onSelectEvidence(ev, run.id)}
                    />
                    <p className="text-xs text-muted">
                      {run.steps.length} tool {run.steps.length === 1 ? "call" : "calls"}
                      {run.latencyMs ? ` in ${formatMs(run.latencyMs)}` : ""}
                      {run.model ? `, planned by ${run.model}` : ""}
                    </p>
                  </div>
                )}
                {run.error && !run.answer && (
                  <p className="rounded-lg border border-danger/40 px-3 py-2 text-sm text-danger" role="alert">
                    {run.error}
                  </p>
                )}
              </article>
            ))}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <form
        className="border-t border-line p-3"
        onSubmit={(e) => {
          e.preventDefault();
          submit(draft);
        }}
      >
        <div className="flex items-end gap-2 rounded-lg border border-line bg-bg px-3 py-2 focus-within:border-focus">
          <label htmlFor="question" className="sr-only">
            Question
          </label>
          <textarea
            id="question"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit(draft);
              }
            }}
            rows={1}
            placeholder={hasMedia ? "Ask a question about this media" : "Choose media first"}
            disabled={!hasMedia}
            className="max-h-32 min-h-6 flex-1 resize-none bg-transparent text-[15px] outline-none placeholder:text-muted disabled:cursor-not-allowed"
          />
          {busy ? (
            <Button size="icon" variant="outline" onClick={onStop} aria-label="Stop">
              <Square />
            </Button>
          ) : (
            <Button type="submit" size="icon" variant="primary" disabled={!draft.trim() || !hasMedia} aria-label="Ask">
              <ArrowUp />
            </Button>
          )}
        </div>
      </form>
    </section>
  );
}
