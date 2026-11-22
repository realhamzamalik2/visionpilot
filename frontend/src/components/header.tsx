"use client";

import { FolderGit2, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import { getHealth, type Health } from "@/lib/api";
import { cn } from "@/lib/utils";

const REPO_URL = process.env.NEXT_PUBLIC_REPO_URL ?? "https://github.com/realhamzamalik2/visionpilot";

export function Header() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await getHealth();
        if (alive) {
          setHealth(h);
          setOffline(false);
        }
      } catch {
        if (alive) setOffline(true);
      }
    };
    void poll();
    const id = setInterval(poll, 10_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const toggleTheme = () => {
    const dark = !document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("vp-theme", dark ? "dark" : "light");
    } catch {
      /* private mode */
    }
  };

  const ready = !offline && health?.mcp_connected && health.gemini_key_set;
  const status = offline
    ? "API offline"
    : !health
      ? "Connecting"
      : !health.mcp_connected
        ? "Vision tools offline"
        : !health.gemini_key_set
          ? "Gemini key missing"
          : `${health.tools.length} vision tools ready`;
  const detail = offline
    ? "Start the backend: uv run visionpilot-api"
    : health && !health.mcp_connected
      ? "Start the MCP server: uv run visionpilot-mcp --transport streamable-http"
      : health && !health.gemini_key_set
        ? "Add GEMINI_API_KEY to .env and restart the API"
        : health
          ? `Planner: ${health.model}. Tools: ${health.tools.join(", ")}`
          : "Checking the backend";

  return (
    <header className="flex items-center justify-between gap-4 border-b border-line px-5 py-3">
      <div className="flex items-center gap-3">
        <Logo />
        <div>
          <h1 className="text-[17px] font-semibold leading-tight">VisionPilot</h1>
          <p className="hidden text-xs text-muted sm:block">An agent that chooses vision models to answer your question</p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Tooltip content={detail}>
          <span className="flex items-center gap-2 whitespace-nowrap rounded-full border border-line px-3 py-1 text-xs" tabIndex={0}>
            <span className={cn("size-2 rounded-full", ready ? "bg-ok" : offline ? "bg-danger" : "bg-focus")} aria-hidden />
            {status}
          </span>
        </Tooltip>
        <Tooltip content="Source code">
          <a href={REPO_URL} target="_blank" rel="noreferrer" aria-label="Source code on GitHub" className="inline-flex size-9 items-center justify-center rounded-md text-muted hover:bg-panel-2 hover:text-text">
            <FolderGit2 className="size-4" />
          </a>
        </Tooltip>
        <Button size="icon" variant="ghost" onClick={toggleTheme} aria-label="Switch light or dark theme">
          <Sun className="hidden dark:block" />
          <Moon className="dark:hidden" />
        </Button>
      </div>
    </header>
  );
}

/** A lens aperture inside viewfinder brackets. */
function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="size-8" aria-hidden>
      <path d="M3 10V5a2 2 0 0 1 2-2h5M22 3h5a2 2 0 0 1 2 2v5M29 22v5a2 2 0 0 1-2 2h-5M10 29H5a2 2 0 0 1-2-2v-5" fill="none" stroke="var(--focus)" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="16" cy="16" r="7" fill="none" stroke="var(--text)" strokeWidth="2.2" />
      <circle cx="16" cy="16" r="2.4" fill="var(--focus)" />
    </svg>
  );
}
