"use client";

import { Check, ChevronRight, CircleAlert, Loader2 } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import type { Run, ToolStep } from "@/hooks/use-agent";
import { cn, formatMs, toolName } from "@/lib/utils";

/** The agent's steps as they stream in: one numbered row per tool call. */
export function Trace({ run }: { run: Run }) {
  return (
    <ol className="flex flex-col gap-1.5" aria-label="Agent steps">
      <AnimatePresence initial={false}>
        {run.steps.map((step, i) => (
          <motion.li
            key={step.callId}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
          >
            <StepRow step={step} index={i + 1} />
          </motion.li>
        ))}
      </AnimatePresence>
      {run.status === "running" && run.thinking && (
        <li className="flex items-center gap-2 pl-1 text-sm text-muted" aria-live="polite">
          <Loader2 className="size-3.5 animate-spin" />
          {run.thinking === "Wrapping up" ? "Writing the answer" : run.steps.length ? "Deciding the next step" : "Planning which tools to use"}
        </li>
      )}
    </ol>
  );
}

function StepRow({ step, index }: { step: ToolStep; index: number }) {
  const args = Object.entries(step.args).filter(([, v]) => v !== null && v !== undefined);
  return (
    <details className="group rounded-lg border border-line bg-panel open:bg-panel">
      <summary className="flex cursor-pointer list-none items-start gap-3 px-3 py-2.5 [&::-webkit-details-marker]:hidden">
        <span
          className={cn(
            "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold",
            step.status === "running" && "bg-focus-soft text-focus",
            step.status === "ok" && "bg-panel-2 text-ok",
            step.status === "error" && "bg-panel-2 text-danger",
          )}
          aria-label={step.status === "running" ? "Running" : step.status === "ok" ? "Done" : "Failed"}
        >
          {step.status === "running" ? <Loader2 className="size-3 animate-spin" /> : step.status === "ok" ? <Check className="size-3" /> : <CircleAlert className="size-3" />}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-baseline justify-between gap-2">
            <span className="text-sm font-medium">
              <span className="mr-1.5 text-muted">{index}.</span>
              {toolName(step.tool)}
            </span>
            {step.latencyMs != null && <span className="shrink-0 text-xs text-muted">{formatMs(step.latencyMs)}</span>}
          </span>
          <span className={cn("block text-sm", step.status === "error" ? "text-danger" : "text-muted")}>
            {step.status === "running" ? "Running…" : step.summary}
          </span>
        </span>
        <ChevronRight className="mt-1 size-4 shrink-0 text-muted transition-transform group-open:rotate-90" />
      </summary>
      <div className="border-t border-line px-3 py-2 text-xs">
        <p className="mb-1 text-muted">
          Tool <code>{step.tool}</code>
          {args.length ? " with " : ""}
          {args.map(([k, v]) => (
            <code key={k} className="mr-1 rounded bg-panel-2 px-1">
              {k}={JSON.stringify(v)}
            </code>
          ))}
        </p>
        {step.data && (
          <pre className="max-h-56 overflow-auto rounded bg-stage p-2 text-[11px] leading-snug">
            {JSON.stringify(trimForDisplay(step.data), null, 2)}
          </pre>
        )}
      </div>
    </details>
  );
}

function trimForDisplay(data: Record<string, unknown>): unknown {
  return JSON.parse(
    JSON.stringify(data, (key, value) => {
      if (key === "depth_png") return "(depth image)";
      if (key === "polygon" || key === "keypoints" || key === "boxes") return Array.isArray(value) ? `(${value.length} points)` : value;
      return value;
    }),
  );
}
