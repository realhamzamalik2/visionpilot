"""Logging that is safe for MCP stdio (never writes to stdout) plus a JSONL latency log."""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_metrics_lock = threading.Lock()
_metrics_path: Path | None = None
_configured = False

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str, log_dir: Path) -> None:
    """Send logs to stderr and logs/mcp_server.log. stdout is reserved for the MCP stdio protocol."""
    global _metrics_path, _configured
    if _configured:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    _metrics_path = log_dir / "tool_metrics.jsonl"

    root = logging.getLogger()
    root.setLevel(level.upper())
    formatter = logging.Formatter(LOG_FORMAT)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(log_dir / "mcp_server.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.handlers = [stderr_handler, file_handler]

    # Third-party libraries are chatty at INFO.
    for noisy in ("httpx", "urllib3", "transformers", "ultralytics", "RapidOCR"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def record_tool_metric(**fields: Any) -> None:
    """Append one JSON line per tool call. Used to build the latency table in the README."""
    if _metrics_path is None:
        return
    fields = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **fields}
    line = json.dumps(fields, ensure_ascii=False)
    with _metrics_lock, _metrics_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
