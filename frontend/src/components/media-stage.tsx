"use client";

import { ImageUp } from "lucide-react";
import { type RefObject, useEffect, useMemo, useRef, useState } from "react";

import { OverlayLayer } from "@/components/overlay-layer";
import { Button } from "@/components/ui/button";
import { VideoTimeline } from "@/components/video-timeline";
import { type MediaInfo, mediaSrc } from "@/lib/api";
import { LAYER_LABELS, type LayerId, type OverlayModel, videoFrameItems } from "@/lib/overlays";
import { cn, formatBytes, toolName } from "@/lib/utils";

interface Props {
  media: MediaInfo;
  model: OverlayModel;
  highlighted: Set<string>;
  activeTool: string | null;
  running: boolean;
  onChangeMedia: () => void;
  seekRef: RefObject<((t: number) => void) | null>;
}

const LAYER_ORDER: LayerId[] = ["boxes", "masks", "pose", "text", "depth"];

export function MediaStage({ media, model, highlighted, activeTool, running, onChangeMedia, seekRef }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(
    media.width && media.height ? { w: media.width, h: media.height } : null,
  );
  // Depth is opt-in: it would otherwise cover the boxes the answer refers to.
  const [hidden, setHidden] = useState<Set<LayerId>>(new Set(["depth"]));
  const [hovered, setHovered] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const frameRef = useRef<HTMLDivElement>(null);
  const [shownWidth, setShownWidth] = useState(0);

  useEffect(() => {
    const el = frameRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setShownWidth(entry.contentRect.width));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    seekRef.current = (t: number) => {
      const v = videoRef.current;
      if (!v) return;
      v.pause();
      v.currentTime = t;
      setTime(t);
    };
    return () => {
      seekRef.current = null;
    };
  }, [seekRef]);

  const visible = useMemo(() => new Set(LAYER_ORDER.filter((l) => !hidden.has(l))), [hidden]);
  const frame = model.video ? videoFrameItems(model.video, time) : null;
  const items = media.kind === "video" ? (frame?.items ?? []) : model.items;
  const litFrames = useMemo(() => {
    const set = new Set<number>();
    if (!model.video) return set;
    highlighted.forEach((k) => {
      const [step, idx] = k.split(":").map(Number);
      if (step === model.video!.step) set.add(idx);
    });
    return set;
  }, [highlighted, model.video]);

  const w = dims?.w ?? model.video?.width ?? 16;
  const h = dims?.h ?? model.video?.height ?? 9;
  const presentLayers = LAYER_ORDER.filter((l) => model.layers.has(l));

  return (
    <section className="flex min-h-0 flex-col gap-3" aria-label="Media">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold">{media.filename}</h2>
          <p className="text-xs text-muted">
            {media.kind === "video" ? "Video" : "Image"}
            {dims ? `, ${dims.w}×${dims.h}` : ""}, {formatBytes(media.size_bytes)}
          </p>
        </div>
        <Button size="sm" variant="ghost" onClick={onChangeMedia}>
          <ImageUp /> Change media
        </Button>
      </div>

      {/* The viewfinder: corner brackets turn amber and pulse while the agent is working. */}
      <div className="relative flex min-h-[280px] flex-1 items-center justify-center rounded-xl bg-stage px-5 pb-11 pt-5">
        <Viewfinder active={running} />
        <div
          ref={frameRef}
          className="relative overflow-hidden rounded-md"
          style={{ width: `min(100%, calc(${model.video ? 50 : 66}vh * ${w / h}))`, aspectRatio: `${w} / ${h}` }}
        >
          {media.kind === "image" ? (
            // eslint-disable-next-line @next/next/no-img-element -- served by our API, sized by the overlay box
            <img
              src={mediaSrc(media)}
              alt={media.filename}
              className="size-full object-fill"
              onLoad={(e) => setDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
            />
          ) : (
            <video
              ref={videoRef}
              src={mediaSrc(media)}
              className="size-full object-fill"
              controls
              preload="metadata"
              onLoadedMetadata={(e) => setDims({ w: e.currentTarget.videoWidth, h: e.currentTarget.videoHeight })}
              onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
            />
          )}
          {dims && (
            <div className={cn(media.kind === "video" && "pointer-events-none")}>
              <OverlayLayer
                scale={shownWidth ? dims.w / shownWidth : 1}
                width={dims.w}
                height={dims.h}
                items={items}
                skeleton={model.skeleton}
                depthPng={model.depthPng}
                visible={visible}
                highlighted={highlighted}
                hovered={hovered}
                onHover={setHovered}
              />
            </div>
          )}
        </div>
        {running && (
          <p className="absolute bottom-2 left-1/2 -translate-x-1/2 rounded-full bg-panel/90 px-3 py-1 text-xs text-text shadow" aria-live="polite">
            {activeTool ? `Running ${toolName(activeTool).toLowerCase()}…` : "Planning…"}
          </p>
        )}
      </div>

      {presentLayers.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Overlay layers">
          <span className="mr-1 text-xs text-muted">Show</span>
          {presentLayers.map((layer) => {
            const on = !hidden.has(layer);
            return (
              <button
                key={layer}
                aria-pressed={on}
                onClick={() =>
                  setHidden((prev) => {
                    const next = new Set(prev);
                    if (on) next.add(layer);
                    else next.delete(layer);
                    return next;
                  })
                }
                className={cn(
                  "rounded-full border px-3 py-1 text-xs transition-colors",
                  on ? "border-line bg-panel text-text" : "border-transparent text-muted line-through",
                )}
              >
                {LAYER_LABELS[layer]}
              </button>
            );
          })}
        </div>
      )}

      {model.video && model.video.timeline.length > 0 && (
        <VideoTimeline video={model.video} currentTime={time} litFrames={litFrames} onSeek={(t) => seekRef.current?.(t)} />
      )}
    </section>
  );
}

function Viewfinder({ active }: { active: boolean }) {
  const corner = "absolute size-6 border-[2.5px] transition-colors duration-300";
  const color = active ? "border-focus vp-pulse" : "border-muted/40";
  return (
    <div className="pointer-events-none absolute inset-2" aria-hidden>
      <span className={cn(corner, color, "left-0 top-0 rounded-tl-md border-b-0 border-r-0")} />
      <span className={cn(corner, color, "right-0 top-0 rounded-tr-md border-b-0 border-l-0")} />
      <span className={cn(corner, color, "bottom-0 left-0 rounded-bl-md border-r-0 border-t-0")} />
      <span className={cn(corner, color, "bottom-0 right-0 rounded-br-md border-l-0 border-t-0")} />
    </div>
  );
}
