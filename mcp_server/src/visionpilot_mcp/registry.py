"""Lazy model loading under a RAM budget (8 GB laptop): load on first use, evict least recently used."""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from visionpilot_mcp.errors import VisionToolError
from visionpilot_mcp.schemas import LoadedModelInfo

logger = logging.getLogger(__name__)

Loader = Callable[[], Any]


@dataclass
class _Entry:
    loader: Loader
    cost_mb: int


class ModelRegistry:
    """Maps a model name to a loader plus its estimated RAM cost.

    Models load on first use. Before loading, the least recently used models are evicted until the new one
    fits inside `budget_mb`. A model that is larger than the whole budget still loads (alone).
    """

    def __init__(self, budget_mb: int) -> None:
        self.budget_mb = budget_mb
        self._entries: dict[str, _Entry] = {}
        self._loaded: OrderedDict[str, Any] = OrderedDict()
        self._load_seconds: dict[str, float] = {}
        self._lock = threading.RLock()
        self._load_lock = threading.Lock()

    def register(self, name: str, loader: Loader, cost_mb: int = 100) -> None:
        with self._lock:
            self._entries[name] = _Entry(loader=loader, cost_mb=cost_mb)
            self._loaded.pop(name, None)

    @property
    def available(self) -> list[str]:
        return sorted(self._entries)

    @property
    def used_mb(self) -> int:
        with self._lock:
            return sum(self._entries[n].cost_mb for n in self._loaded)

    def get(self, name: str) -> Any:
        with self._lock:  # fast path: already loaded
            if name in self._loaded:
                self._loaded.move_to_end(name)
                return self._loaded[name]
            entry = self._entries.get(name)
            if entry is None:
                raise VisionToolError(f"Unknown model '{name}'. Available: {self.available}")

        # Loads are serialized (two heavy loads at once could exhaust RAM), but the bookkeeping lock is not
        # held while loading, so tools whose models are already in memory keep running.
        with self._load_lock:
            with self._lock:
                if name in self._loaded:  # another thread loaded it while we waited
                    self._loaded.move_to_end(name)
                    return self._loaded[name]
                while self._loaded and self.used_mb + entry.cost_mb > self.budget_mb:
                    self._evict_oldest()

            start = time.perf_counter()
            logger.info("Loading model %s (~%d MB) ...", name, entry.cost_mb)
            model = entry.loader()
            elapsed = time.perf_counter() - start
            with self._lock:
                self._loaded[name] = model
                self._load_seconds[name] = round(elapsed, 2)
            logger.info("Loaded model %s in %.2fs (in use: %d/%d MB)", name, elapsed, self.used_mb, self.budget_mb)
            return model

    def warm_up(self, names: Iterable[str], *, background: bool = True) -> threading.Thread | None:
        """Pre-load models so the first user question is fast. Failures are logged, never raised."""
        names = [n for n in names if n in self._entries]

        def run() -> None:
            for name in names:
                try:
                    self.get(name)
                except Exception:
                    logger.exception("Warm-up of %s failed (it will be retried on first use)", name)

        if not background:
            run()
            return None
        thread = threading.Thread(target=run, name="model-warmup", daemon=True)
        thread.start()
        return thread

    def unload(self, name: str) -> bool:
        with self._lock:
            removed = self._loaded.pop(name, None) is not None
        if removed:
            gc.collect()
        return removed

    def status(self) -> list[LoadedModelInfo]:
        with self._lock:
            return [
                LoadedModelInfo(
                    name=n, load_seconds=self._load_seconds.get(n, 0.0), estimated_mb=self._entries[n].cost_mb
                )
                for n in self._loaded
            ]

    def _evict_oldest(self) -> None:
        name, _ = self._loaded.popitem(last=False)
        logger.info("Evicting model %s to stay within %d MB", name, self.budget_mb)
        gc.collect()
