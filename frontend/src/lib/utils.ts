import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Distinguishable on both navy and white, and none of them amber (amber means "highlighted").
const LABEL_COLORS = ["#5B9BF0", "#3FBFA8", "#C987E8", "#E8798F", "#8FB65A", "#56C3E8", "#B79CF2", "#E0A06A"];

export function labelColor(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
  return LABEL_COLORS[hash % LABEL_COLORS.length];
}

export function formatMs(ms?: number | null): string {
  if (ms == null) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export const TOOL_NAMES: Record<string, string> = {
  detect_objects: "Detect objects",
  segment_objects: "Segment objects",
  estimate_pose: "Estimate pose",
  estimate_depth: "Estimate depth",
  describe_image: "Describe scene",
  find_phrase: "Find phrase",
  read_text: "Read text",
  read_korean_text: "Read Korean text",
  analyze_faces: "Analyze faces",
  process_video: "Analyze video",
};

export const toolName = (tool?: string) => (tool ? (TOOL_NAMES[tool] ?? tool) : "Tool");
