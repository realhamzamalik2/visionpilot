/** Turns raw tool results into drawable overlay items, keyed so answer evidence can highlight them. */

export type LayerId = "boxes" | "masks" | "pose" | "text" | "depth";

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface OverlayItem {
  key: string; // `${step}:${index}` matches Evidence.step + item_indices
  layer: LayerId;
  label: string;
  confidence?: number;
  box?: Box;
  polygon?: [number, number][];
  keypoints?: { x: number; y: number; confidence: number }[];
}

export interface StepResult {
  step: number;
  tool: string;
  data: Record<string, unknown>;
}

export interface TimelineFrame {
  t: number;
  frame_index: number;
  counts: Record<string, number>;
  total: number;
  boxes: { label: string; confidence: number; box: Box }[];
}

export interface VideoAnalysis {
  step: number;
  duration_s: number;
  width: number;
  height: number;
  task: string;
  timeline: TimelineFrame[];
  stats: { label: string; max: number; mean: number; peak_t: number; last: number }[];
}

export interface OverlayModel {
  items: OverlayItem[];
  skeleton: [number, number][];
  depthPng?: string;
  layers: Set<LayerId>;
  video?: VideoAnalysis;
}

type Row = Record<string, unknown>;
const rows = (value: unknown): Row[] => (Array.isArray(value) ? (value as Row[]) : []);

export function buildOverlays(results: StepResult[]): OverlayModel {
  const model: OverlayModel = { items: [], skeleton: [], layers: new Set() };
  const add = (item: OverlayItem) => {
    model.items.push(item);
    model.layers.add(item.layer);
  };

  for (const { step, tool, data } of results) {
    const key = (i: number) => `${step}:${i}`;
    switch (tool) {
      case "detect_objects":
        rows(data.detections).forEach((d, i) =>
          add({ key: key(i), layer: "boxes", label: String(d.label), confidence: Number(d.confidence), box: d.box as Box }),
        );
        break;
      case "find_phrase":
        rows(data.matches).forEach((m, i) => add({ key: key(i), layer: "boxes", label: String(m.label), box: m.box as Box }));
        break;
      case "analyze_faces":
        rows(data.faces).forEach((f, i) =>
          add({ key: key(i), layer: "boxes", label: String(f.expression), confidence: Number(f.confidence), box: f.box as Box }),
        );
        break;
      case "segment_objects":
        rows(data.objects).forEach((o, i) =>
          add({
            key: key(i),
            layer: "masks",
            label: String(o.label),
            confidence: Number(o.confidence),
            box: o.box as Box,
            polygon: o.polygon as [number, number][],
          }),
        );
        break;
      case "estimate_pose":
        model.skeleton = (data.skeleton as [number, number][]) ?? [];
        rows(data.people).forEach((p, i) =>
          add({
            key: key(i),
            layer: "pose",
            label: String(p.posture ?? "person"),
            confidence: Number(p.confidence),
            box: p.box as Box,
            keypoints: p.keypoints as OverlayItem["keypoints"],
          }),
        );
        break;
      case "read_text":
      case "read_korean_text":
        rows(data.lines).forEach((l, i) =>
          add({ key: key(i), layer: "text", label: String(l.text), box: l.box as Box, polygon: l.polygon as [number, number][] }),
        );
        break;
      case "estimate_depth":
        // Distances measured inside boxes from an earlier step: shown as tagged boxes so evidence can point at them.
        rows(data.regions).forEach((r, i) =>
          add({ key: key(i), layer: "boxes", label: `${Number(r.median_m).toFixed(1)} m`, box: r.box as Box }),
        );
        if (typeof data.depth_png === "string") {
          model.depthPng = data.depth_png;
          model.layers.add("depth");
        }
        break;
      case "process_video":
        model.video = {
          step,
          duration_s: Number(data.duration_s),
          width: Number(data.width),
          height: Number(data.height),
          task: String(data.task),
          timeline: (data.timeline as TimelineFrame[]) ?? [],
          stats: (data.stats as VideoAnalysis["stats"]) ?? [],
        };
        model.layers.add("boxes");
        break;
    }
  }
  return model;
}

/** Boxes for the sampled frame closest to the video's current time (key = `${step}:${frameIndexInTimeline}`). */
export function videoFrameItems(video: VideoAnalysis, currentTime: number): { frameIdx: number; items: OverlayItem[] } {
  if (!video.timeline.length) return { frameIdx: -1, items: [] };
  let best = 0;
  for (let i = 1; i < video.timeline.length; i++) {
    if (Math.abs(video.timeline[i].t - currentTime) < Math.abs(video.timeline[best].t - currentTime)) best = i;
  }
  const frame = video.timeline[best];
  const items = frame.boxes.map((b, j) => ({
    key: `${video.step}:${best}:${j}`,
    layer: "boxes" as const,
    label: b.label,
    confidence: b.confidence,
    box: b.box,
  }));
  return { frameIdx: best, items };
}

export const LAYER_LABELS: Record<LayerId, string> = {
  boxes: "Boxes",
  masks: "Masks",
  pose: "Pose",
  text: "Text",
  depth: "Depth",
};
