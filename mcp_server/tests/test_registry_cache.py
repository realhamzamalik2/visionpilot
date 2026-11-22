from __future__ import annotations

import threading
import time

import pytest

from visionpilot_mcp.cache import ResultCache
from visionpilot_mcp.errors import VisionToolError
from visionpilot_mcp.registry import ModelRegistry
from visionpilot_mcp.schemas import DescribeImageResult, ImageInfo


def test_registry_loads_lazily_and_reuses() -> None:
    loads: list[str] = []
    registry = ModelRegistry(budget_mb=1000)
    registry.register("a", lambda: loads.append("a") or "model-a")

    assert loads == []  # nothing loaded at registration
    assert registry.get("a") == "model-a"
    assert registry.get("a") == "model-a"
    assert loads == ["a"]


def test_registry_evicts_least_recently_used_to_fit_budget() -> None:
    registry = ModelRegistry(budget_mb=300)
    for name in ("a", "b", "c"):
        registry.register(name, lambda n=name: f"model-{n}", cost_mb=120)

    registry.get("a")
    registry.get("b")
    registry.get("a")  # a is now most recent
    registry.get("c")  # 360 MB > 300: evicts b
    assert [m.name for m in registry.status()] == ["a", "c"]
    assert registry.used_mb == 240


def test_model_bigger_than_budget_still_loads_alone() -> None:
    registry = ModelRegistry(budget_mb=100)
    registry.register("small", lambda: "s", cost_mb=50)
    registry.register("huge", lambda: "h", cost_mb=500)
    registry.get("small")
    assert registry.get("huge") == "h"
    assert [m.name for m in registry.status()] == ["huge"]


def test_loaded_models_stay_usable_while_another_loads() -> None:
    registry = ModelRegistry(budget_mb=1000)
    started = threading.Event()

    def slow_loader() -> str:
        started.set()
        time.sleep(0.5)
        return "slow"

    registry.register("fast", lambda: "fast")
    registry.register("slow", slow_loader)
    registry.get("fast")
    registry.warm_up(["slow"])
    started.wait(1)
    t0 = time.perf_counter()
    assert registry.get("fast") == "fast"
    assert time.perf_counter() - t0 < 0.2  # did not wait for the slow load


def test_warm_up_swallows_errors() -> None:
    registry = ModelRegistry(budget_mb=100)
    registry.register("broken", lambda: 1 / 0)
    registry.warm_up(["broken", "not-registered"], background=False)
    assert registry.status() == []


def test_registry_unknown_model() -> None:
    with pytest.raises(VisionToolError, match="Unknown model"):
        ModelRegistry(budget_mb=100).get("nope")


def _result(caption: str) -> DescribeImageResult:
    info = ImageInfo(width=1, height=1, sha1="abc", source_type="file")
    return DescribeImageResult(image=info, model="m", detail="brief", caption=caption)


def test_cache_marks_hits_and_respects_size() -> None:
    cache = ResultCache(maxsize=2)
    k1 = ResultCache.make_key("t", "img1", {"x": 1})
    k2 = ResultCache.make_key("t", "img2", {"x": 1})
    k3 = ResultCache.make_key("t", "img3", {"x": 1})

    assert cache.get(k1, DescribeImageResult) is None
    cache.put(k1, _result("one"))
    hit = cache.get(k1, DescribeImageResult)
    assert hit is not None and hit.cached and hit.caption == "one"

    cache.put(k2, _result("two"))
    cache.put(k3, _result("three"))  # evicts k1
    assert cache.get(k1, DescribeImageResult) is None
    assert len(cache) == 2
    assert (cache.hits, cache.misses) == (1, 2)


def test_cache_key_ignores_param_order() -> None:
    assert ResultCache.make_key("t", "i", {"a": 1, "b": 2}) == ResultCache.make_key("t", "i", {"b": 2, "a": 1})
