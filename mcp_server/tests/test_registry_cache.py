from __future__ import annotations

import pytest

from visionpilot_mcp.cache import ResultCache
from visionpilot_mcp.errors import VisionToolError
from visionpilot_mcp.registry import ModelRegistry
from visionpilot_mcp.schemas import DescribeImageResult, ImageInfo


def test_registry_loads_lazily_and_reuses() -> None:
    loads: list[str] = []
    registry = ModelRegistry(max_loaded=2)
    registry.register("a", lambda: loads.append("a") or "model-a")

    assert loads == []  # nothing loaded at registration
    assert registry.get("a") == "model-a"
    assert registry.get("a") == "model-a"
    assert loads == ["a"]


def test_registry_evicts_least_recently_used() -> None:
    registry = ModelRegistry(max_loaded=2)
    for name in ("a", "b", "c"):
        registry.register(name, lambda n=name: f"model-{n}")

    registry.get("a")
    registry.get("b")
    registry.get("a")  # a is now most recent
    registry.get("c")  # evicts b
    assert [m.name for m in registry.status()] == ["a", "c"]


def test_registry_unknown_model() -> None:
    with pytest.raises(VisionToolError, match="Unknown model"):
        ModelRegistry(max_loaded=1).get("nope")


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
