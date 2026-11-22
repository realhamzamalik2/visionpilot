"""In-memory LRU cache of tool results, keyed by tool name, image hash and arguments."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

ResultT = TypeVar("ResultT", bound=BaseModel)


class ResultCache:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._items: OrderedDict[str, BaseModel] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(tool: str, image_sha1: str, params: Mapping[str, Any]) -> str:
        return f"{tool}:{image_sha1}:{json.dumps(params, sort_keys=True, default=str)}"

    def get(self, key: str, result_type: type[ResultT]) -> ResultT | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
        if not isinstance(item, result_type):
            return None
        return item.model_copy(update={"cached": True})

    def put(self, key: str, value: BaseModel) -> None:
        if self.maxsize == 0:
            return
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)
