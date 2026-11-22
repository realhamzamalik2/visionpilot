"use client";

import { MotionConfig } from "motion/react";
import { useCallback, useMemo, useRef, useState } from "react";

import { ChatPanel } from "@/components/chat-panel";
import { Header } from "@/components/header";
import { MediaPicker } from "@/components/media-picker";
import { MediaStage } from "@/components/media-stage";
import { TooltipProvider } from "@/components/ui/tooltip";
import { stepResults, useAgent } from "@/hooks/use-agent";
import type { Evidence, MediaInfo } from "@/lib/api";
import { buildOverlays } from "@/lib/overlays";

export function Workbench() {
  const [media, setMedia] = useState<MediaInfo | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<Set<string>>(new Set());
  const seekRef = useRef<((t: number) => void) | null>(null);
  const { runs, ask, stop, busy } = useAgent();

  const mediaRuns = useMemo(() => runs.filter((r) => r.mediaId === media?.id), [runs, media?.id]);
  const activeRun = mediaRuns.find((r) => r.id === selectedRunId) ?? mediaRuns.at(-1);
  const model = useMemo(() => buildOverlays(stepResults(activeRun)), [activeRun]);
  const runningStep = activeRun?.steps.findLast((s) => s.status === "running");

  const evidenceKeys = (ev: Evidence) => new Set(ev.item_indices.map((i) => `${ev.step}:${i}`));

  const onHighlight = useCallback(
    (ev: Evidence | null, runId: string) => {
      if (ev) setSelectedRunId(runId);
      setHighlight(ev ? evidenceKeys(ev) : new Set());
    },
    [],
  );

  const onSelectEvidence = useCallback(
    (ev: Evidence, runId: string) => {
      setSelectedRunId(runId);
      // For a video, evidence points at timeline frames: jump to the first one.
      const run = runs.find((r) => r.id === runId);
      const step = run?.steps.find((s) => s.step === ev.step && s.tool === "process_video");
      const timeline = step?.data?.timeline as { t: number }[] | undefined;
      const first = ev.item_indices[0];
      if (timeline && first != null && timeline[first]) seekRef.current?.(timeline[first].t);
    },
    [runs],
  );

  const pick = (next: MediaInfo, nextSuggestions: string[]) => {
    stop();
    setMedia(next);
    setSuggestions(nextSuggestions);
    setSelectedRunId(null);
    setHighlight(new Set());
  };

  return (
    <MotionConfig reducedMotion="user">
      <TooltipProvider>
        <div className="flex min-h-dvh flex-col lg:h-dvh">
          <Header />
          <main className="grid min-h-0 flex-1 gap-4 p-4 lg:grid-cols-[minmax(0,1.55fr)_minmax(360px,1fr)]">
            <div className="min-h-0 overflow-y-auto">
              {media ? (
                <MediaStage
                  key={media.id}
                  media={media}
                  model={model}
                  highlighted={highlight}
                  activeTool={runningStep?.tool ?? null}
                  running={activeRun?.status === "running"}
                  onChangeMedia={() => {
                    stop();
                    setMedia(null);
                  }}
                  seekRef={seekRef}
                />
              ) : (
                <MediaPicker onPick={pick} />
              )}
            </div>
            <div className="min-h-[420px] lg:min-h-0">
              <ChatPanel
                runs={mediaRuns}
                activeRunId={activeRun?.id ?? null}
                onSelectRun={setSelectedRunId}
                suggestions={suggestions}
                hasMedia={!!media}
                busy={busy}
                onAsk={(q) => {
                  if (!media) return;
                  setSelectedRunId(null);
                  void ask(media, q);
                }}
                onStop={stop}
                onHighlight={onHighlight}
                onSelectEvidence={onSelectEvidence}
              />
            </div>
          </main>
        </div>
      </TooltipProvider>
    </MotionConfig>
  );
}
