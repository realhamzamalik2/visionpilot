"use client";

import type { LayerId, OverlayItem } from "@/lib/overlays";
import { labelColor } from "@/lib/utils";

interface Props {
  /** Image pixels per screen pixel, so labels and lines keep the same on-screen size for any photo size. */
  scale: number;
  width: number;
  height: number;
  items: OverlayItem[];
  skeleton: [number, number][];
  depthPng?: string;
  visible: Set<LayerId>;
  highlighted: Set<string>;
  hovered: string | null;
  onHover: (key: string | null) => void;
}

const FOCUS = "var(--focus)";

/** Draws boxes, masks, skeletons and text regions in image-pixel coordinates (viewBox = image size). */
export function OverlayLayer({ scale, width, height, items, skeleton, depthPng, visible, highlighted, hovered, onHover }: Props) {
  const stroke = 1.75 * scale;
  const font = 12.5 * scale;
  const isLit = (key: string) => highlighted.has(key) || hovered === key || [...highlighted].some((k) => key.startsWith(`${k}:`));
  // Only dim the rest when something on screen actually matches (evidence can point at non-visual results).
  const anyHighlight = items.some((item) => visible.has(item.layer) && isLit(item.key));

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="absolute inset-0 size-full"
      role="img"
      aria-label="Detection overlays"
    >
      {depthPng && visible.has("depth") && (
        <image href={depthPng} x={0} y={0} width={width} height={height} preserveAspectRatio="none" opacity={0.72} />
      )}

      {items
        .filter((item) => visible.has(item.layer))
        .map((item) => {
          const lit = isLit(item.key);
          const color = lit ? FOCUS : labelColor(item.label);
          const dim = anyHighlight && !lit ? 0.25 : 1;
          const box = item.box;
          return (
            <g
              key={item.key}
              opacity={dim}
              onMouseEnter={() => onHover(item.key)}
              onMouseLeave={() => onHover(null)}
              style={{ transition: "opacity 150ms" }}
            >
              {item.polygon && item.polygon.length > 2 && (
                <polygon
                  points={item.polygon.map(([x, y]) => `${x},${y}`).join(" ")}
                  fill={color}
                  fillOpacity={item.layer === "masks" ? (lit ? 0.42 : 0.28) : 0.08}
                  stroke={color}
                  strokeWidth={stroke}
                  strokeLinejoin="round"
                />
              )}

              {item.layer === "pose" && item.keypoints && (
                <g>
                  {skeleton.map(([a, b], i) => {
                    const p = item.keypoints![a];
                    const q = item.keypoints![b];
                    if (!p || !q || p.confidence < 0.3 || q.confidence < 0.3) return null;
                    return (
                      <line key={i} x1={p.x} y1={p.y} x2={q.x} y2={q.y} stroke={color} strokeWidth={stroke * 1.6} strokeLinecap="round" />
                    );
                  })}
                  {item.keypoints.map((p, i) =>
                    p.confidence >= 0.3 ? <circle key={i} cx={p.x} cy={p.y} r={stroke * 2} fill={color} /> : null,
                  )}
                </g>
              )}

              {box && item.layer !== "pose" && !(item.layer === "text" && item.polygon) && (
                <rect
                  x={box.x1}
                  y={box.y1}
                  width={Math.max(box.x2 - box.x1, 1)}
                  height={Math.max(box.y2 - box.y1, 1)}
                  fill={lit ? "var(--focus-soft)" : "transparent"}
                  stroke={color}
                  strokeWidth={lit ? stroke * 2 : stroke}
                  rx={stroke * 1.5}
                />
              )}

              {box && item.layer !== "text" && (
                <Tag x={box.x1} y={box.y1} font={font} color={color} lit={lit}>
                  {item.label}
                  {item.confidence != null && !Number.isNaN(item.confidence) ? ` ${Math.round(item.confidence * 100)}%` : ""}
                </Tag>
              )}
            </g>
          );
        })}
    </svg>
  );
}

function Tag({ x, y, font, color, lit, children }: { x: number; y: number; font: number; color: string; lit: boolean; children: React.ReactNode }) {
  const text = String(Array.isArray(children) ? children.join("") : children);
  const w = text.length * font * 0.56 + font * 0.8;
  const h = font * 1.45;
  const ty = y - h < 0 ? y : y - h;
  return (
    <g pointerEvents="none">
      <rect x={x} y={ty} width={w} height={h} rx={font * 0.25} fill={color} />
      <text x={x + font * 0.4} y={ty + h * 0.72} fontSize={font} fontWeight={600} fill={lit ? "#14213a" : "#0f1a2e"}>
        {text}
      </text>
    </g>
  );
}
