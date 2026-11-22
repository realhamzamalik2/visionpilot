"use client";

import { useMemo, useState } from "react";

import type { VideoAnalysis } from "@/lib/overlays";
import { cn } from "@/lib/utils";

interface Props {
  video: VideoAnalysis;
  currentTime: number;
  litFrames: Set<number>;
  onSeek: (t: number) => void;
}

const W = 1000;
const H = 120;
const PAD = { top: 12, right: 12, bottom: 22, left: 30 };

/** Count-over-time chart under the player. Click anywhere to jump the video to that moment. */
export function VideoTimeline({ video, currentTime, litFrames, onSeek }: Props) {
  const labels = useMemo(() => video.stats.map((s) => s.label), [video.stats]);
  const [label, setLabel] = useState<string>(labels[0] ?? "");
  const [hover, setHover] = useState<number | null>(null);

  const duration = Math.max(video.duration_s, video.timeline.at(-1)?.t ?? 0, 0.001);
  const values = video.timeline.map((f) => f.counts[label] ?? 0);
  const max = Math.max(1, ...values);
  const x = (t: number) => PAD.left + (t / duration) * (W - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (1 - v / max) * (H - PAD.top - PAD.bottom);
  const points = video.timeline.map((f, i) => [x(f.t), y(values[i])] as const);
  const line = points.map(([px, py], i) => `${i ? "L" : "M"}${px},${py}`).join(" ");
  const area = points.length ? `${line} L${points.at(-1)![0]},${y(0)} L${points[0][0]},${y(0)} Z` : "";

  const seekFromEvent = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * W;
    const t = ((px - PAD.left) / (W - PAD.left - PAD.right)) * duration;
    return Math.min(Math.max(t, 0), duration);
  };
  const nearest = (t: number) =>
    video.timeline.reduce((best, f, i) => (Math.abs(f.t - t) < Math.abs(video.timeline[best].t - t) ? i : best), 0);

  const hoverFrame = hover != null ? video.timeline[hover] : null;
  const stat = video.stats.find((s) => s.label === label);

  return (
    <div className="rounded-lg border border-line bg-panel p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-sm">
        <div className="flex items-baseline gap-2">
          <span className="font-semibold">{label ? `${label} count over time` : "Nothing detected"}</span>
          {stat && (
            <span className="text-muted">
              peak {stat.max} at {stat.peak_t.toFixed(1)}s, average {stat.mean.toFixed(1)}
            </span>
          )}
        </div>
        {labels.length > 1 && (
          <div className="flex gap-1" role="radiogroup" aria-label="Series">
            {labels.slice(0, 5).map((l) => (
              <button
                key={l}
                role="radio"
                aria-checked={l === label}
                onClick={() => setLabel(l)}
                className={cn(
                  "rounded px-2 py-0.5 text-xs",
                  l === label ? "bg-panel-2 text-text" : "text-muted hover:text-text",
                )}
              >
                {l}
              </button>
            ))}
          </div>
        )}
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-28 w-full cursor-crosshair select-none"
        onClick={(e) => onSeek(video.timeline[nearest(seekFromEvent(e))]?.t ?? 0)}
        onMouseMove={(e) => setHover(nearest(seekFromEvent(e)))}
        onMouseLeave={() => setHover(null)}
        role="slider"
        aria-label="Video timeline"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(currentTime)}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") onSeek(Math.min(currentTime + 1, duration));
          if (e.key === "ArrowLeft") onSeek(Math.max(currentTime - 1, 0));
        }}
      >
        {[0, max].map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(v)} y2={y(v)} stroke="var(--line)" strokeWidth={1} />
            <text x={PAD.left - 6} y={y(v) + 4} textAnchor="end" fontSize={11} fill="var(--muted)">
              {v}
            </text>
          </g>
        ))}
        <text x={PAD.left} y={H - 4} fontSize={11} fill="var(--muted)">
          0s
        </text>
        <text x={W - PAD.right} y={H - 4} fontSize={11} fill="var(--muted)" textAnchor="end">
          {duration.toFixed(0)}s
        </text>

        <path d={area} fill="var(--focus-soft)" />
        <path d={line} fill="none" stroke="var(--muted)" strokeWidth={2} strokeLinejoin="round" />
        {points.map(([px, py], i) =>
          litFrames.has(i) ? <circle key={i} cx={px} cy={py} r={5} fill="var(--focus)" stroke="var(--panel)" strokeWidth={2} /> : null,
        )}

        <line x1={x(currentTime)} x2={x(currentTime)} y1={PAD.top - 4} y2={y(0)} stroke="var(--focus)" strokeWidth={2} />

        {hoverFrame && (
          <g pointerEvents="none">
            <line x1={x(hoverFrame.t)} x2={x(hoverFrame.t)} y1={PAD.top} y2={y(0)} stroke="var(--text)" strokeOpacity={0.3} strokeDasharray="3 3" />
            <text x={Math.min(x(hoverFrame.t) + 6, W - 110)} y={PAD.top + 10} fontSize={12} fill="var(--text)">
              {hoverFrame.t.toFixed(1)}s: {hoverFrame.counts[label] ?? 0} {label}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}
