from __future__ import annotations

from visionpilot_api.config import REPO_ROOT
from visionpilot_api.evals import EvalResult, check_answer, check_tools, load_cases, summarize


def test_question_file_is_valid() -> None:
    cases = load_cases(REPO_ROOT / "evals" / "questions.jsonl")
    assert len(cases) == 25
    assert len({c.id for c in cases}) == 25
    for case in cases:
        assert case.media.startswith("sample:")
        assert case.expect.get("tools_all") or case.expect.get("tools_any")


def test_tool_checks() -> None:
    assert check_tools({"tools_all": ["detect_objects", "estimate_depth"]}, ["detect_objects", "estimate_depth"]) == []
    assert check_tools({"tools_all": ["estimate_depth"]}, ["detect_objects"]) == ["missing tools ['estimate_depth']"]
    assert check_tools({"tools_any": ["find_phrase", "describe_image"]}, ["describe_image"]) == []


def test_answer_checks() -> None:
    assert check_answer({"answer_number": {"min": 4, "max": 4}}, "There are **4 people**.") == []
    assert check_answer({"answer_number": {"min": 4, "max": 4}}, "I see four people.") == []
    assert check_answer({"answer_number": {"min": 6, "max": 6}}, "It costs 6,000 won.") != []  # 6000 is not 6
    assert check_answer({"answer_contains_any": ["9,500", "9500"]}, "Perilla noodles cost 9,500 won") == []
    assert check_answer({"answer_regex": r"\d+(\.\d+)?\s*(m\b|meter)"}, "about 3.2 meters away") == []
    assert check_answer({"answer_contains_all": ["칼국수"]}, "손칼국수 8,000원") == []


def test_summary() -> None:
    results = [
        EvalResult("a", "q", ["detect_objects"], "x", True, True, 10.0),
        EvalResult("b", "q", ["detect_objects", "estimate_depth"], "y", True, False, 20.0),
    ]
    summary = summarize(results, "m")
    assert summary["tool_selection_accuracy"] == 1.0
    assert summary["answer_accuracy"] == 0.5 and summary["task_success"] == 0.5
    assert summary["mean_tool_calls"] == 1.5
