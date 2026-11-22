"use client";

import { useLayoutEffect, useMemo, useRef } from "react";
import ReactMarkdown, { type Components } from "react-markdown";

import type { Evidence, FinalAnswer } from "@/lib/api";

interface Props {
  answer: FinalAnswer;
  onHighlight: (evidence: Evidence | null) => void;
  onSelect?: (evidence: Evidence) => void;
}

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Renders the Markdown answer and turns each evidence label into a hoverable span.
 * Hovering highlights the matching boxes on the image; clicking a video timeline frame seeks to it.
 */
export function Answer({ answer, onHighlight, onSelect }: Props) {
  const { markdown, unmatched } = useMemo(() => {
    let md = answer.answer;
    const missing: number[] = [];
    answer.evidence.forEach((ev, i) => {
      const label = ev.label.trim();
      if (!label) return;
      const re = new RegExp(`(?<![\\[\\w])(${escapeRegExp(label)})(?![\\w\\]])`, "i");
      if (re.test(md)) md = md.replace(re, `[$1](#ev-${i})`);
      else missing.push(i);
    });
    return { markdown: md, unmatched: missing };
  }, [answer]);

  // Handlers live in a ref so the Markdown renderers below stay stable. If the renderer identity changed on
  // every hover, React would remount the evidence buttons between mousedown and mouseup and drop the click.
  const handlers = useRef({ onHighlight, onSelect, answer });
  useLayoutEffect(() => {
    handlers.current = { onHighlight, onSelect, answer };
  });

  const evidenceProps = (index: number) => ({
    onMouseEnter: () => handlers.current.onHighlight(handlers.current.answer.evidence[index]),
    onMouseLeave: () => handlers.current.onHighlight(null),
    onFocus: () => handlers.current.onHighlight(handlers.current.answer.evidence[index]),
    onBlur: () => handlers.current.onHighlight(null),
    onClick: () => handlers.current.onSelect?.(handlers.current.answer.evidence[index]),
  });

  const components = useMemo<Components>(
    () => ({
      a: ({ href, children }) => {
        const match = href?.match(/^#ev-(\d+)$/);
        if (!match) {
          return (
            <a href={href} target="_blank" rel="noreferrer" className="underline">
              {children}
            </a>
          );
        }
        return (
          <button
            type="button"
            {...evidenceProps(Number(match[1]))}
            className="rounded-sm border-b-2 border-dotted border-focus px-0.5 text-inherit hover:bg-focus-soft focus-visible:bg-focus-soft"
          >
            {children}
          </button>
        );
      },
    }),
    [],
  );

  return (
    <div className="prose-answer text-[15px]">
      <ReactMarkdown components={components}>
        {markdown}
      </ReactMarkdown>
      {unmatched.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {unmatched.map((i) => (
            <button
              key={i}
              type="button"
              {...evidenceProps(i)}
              className="rounded-full border border-line px-2.5 py-0.5 text-xs text-muted hover:border-focus hover:text-text"
            >
              {answer.evidence[i].label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
