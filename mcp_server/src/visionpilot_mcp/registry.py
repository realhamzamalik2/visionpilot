"""Lazy model loading with an LRU cap, so only a few models sit in RAM at once (8 GB laptop)."""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from visionpilot_mcp.errors import VisionToolError
from visionpilot_mcp.schemas import LoadedModelInfo

logger = logging.getLogger(__name__)

Loader = Callable[[], Any]


class ModelRegistry:
    """Maps a model name to a loader. Models load on first use and the least recently used one is evicted."""

    def __init__(self, max_loaded: int) -> None:
        self.max_loaded = max_loaded
        self._loaders: dict[str, Loader] = {}
        self._loaded: OrderedDict[str, Any] = OrderedDict()
        self._load_seconds: dict[str, float] = {}
        self._lock = threading.RLock()

    def register(self, name: str, loader: Loader) -> None:
        with self._lock:
            self._loaders[name] = loader
            self._loaded.pop(name, None)

    @property
    def available(self) -> list[str]:
        return sorted(self._loaders)

    def get(self, name: str) -> Any:
        # Loading holds the lock on purpose: two heavy loads at once could exhaust RAM.
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)
                return self._loaded[name]
            loader = self._loaders.get(name)
            if loader is None:
                raise VisionToolError(f"Unknown model '{name}'. Available: {self.available}")

            while len(self._loaded) >= self.max_loaded:
                self._evict_oldest()

            start = time.perf_counter()
            logger.info("Loading model %s ...", name)
            model = loader()
            elapsed = time.perf_counter() - start
            self._loaded[name] = model
            self._load_seconds[name] = round(elapsed, 2)
            logger.info("Loaded model %s in %.2fs", name, elapsed)
            return model

    def unload(self, name: str) -> bool:
        with self._lock:
            removed = self._loaded.pop(name, None) is not None
        if removed:
            gc.collect()
        return removed

    def status(self) -> list[LoadedModelInfo]:
        with self._lock:
            return [LoadedModelInfo(name=n, load_seconds=self._load_seconds.get(n, 0.0)) for n in self._loaded]

    def _evict_oldest(self) -> None:
        name, _ = self._loaded.popitem(last=False)
        logger.info("Evicting model %s to stay within max_loaded_models=%d", name, self.max_loaded)
        gc.collect()
