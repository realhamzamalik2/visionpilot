"""Gemini (free tier) as the planner: function calling, retries with backoff, and a fallback model."""

from __future__ import annotations

import asyncio
import copy
import logging
import random
from typing import Any, Protocol

from google import genai
from google.genai import errors, types

from visionpilot_api.schemas import FinalAnswer, ToolSpec

logger = logging.getLogger(__name__)

RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
FINAL_ANSWER_TOOL = "final_answer"


class Planner(Protocol):
    model_in_use: str

    async def generate(
        self, contents: list[types.Content], tools: list[ToolSpec], system_prompt: str, *, force_answer: bool = False
    ) -> types.GenerateContentResponse: ...


class LLMError(RuntimeError):
    pass


def clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop JSON Schema keys that function-calling APIs often reject or that only add tokens."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):  # inline local refs
                return walk(copy.deepcopy(defs[ref.split("/")[-1]]))
            node.pop("title", None)
            node.pop("$schema", None)
            node.pop("additionalProperties", None)
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    cleaned = walk(schema)
    cleaned.setdefault("type", "object")
    return cleaned


def build_tools(tools: list[ToolSpec]) -> list[types.Tool]:
    declarations = [
        types.FunctionDeclaration(
            name=t.name, description=t.description, parameters_json_schema=clean_schema(t.input_schema)
        )
        for t in tools
    ]
    declarations.append(
        types.FunctionDeclaration(
            name=FINAL_ANSWER_TOOL,
            description=(
                "Finish the task. Call this exactly once with the final answer and the evidence (tool steps and "
                "item indices) that supports each claim."
            ),
            parameters_json_schema=clean_schema(FinalAnswer.model_json_schema()),
        )
    )
    return [types.Tool(function_declarations=declarations)]


class GeminiPlanner:
    def __init__(
        self, api_key: str, model: str, fallback_model: str | None, *, max_retries: int = 3, timeout_s: float = 60.0
    ) -> None:
        if not api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Add your free key from https://aistudio.google.com/apikey to .env"
            )
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self.primary = model
        self.fallback = fallback_model
        self.model_in_use = model
        self.max_retries = max_retries

    async def generate(
        self, contents: list[types.Content], tools: list[ToolSpec], system_prompt: str, *, force_answer: bool = False
    ) -> types.GenerateContentResponse:
        mode = types.FunctionCallingConfigMode.ANY  # always respond with a function call (a tool or final_answer)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=build_tools(tools),  # type: ignore[arg-type]
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=mode, allowed_function_names=[FINAL_ANSWER_TOOL] if force_answer else None
                )
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.2,
        )
        models = [self.model_in_use] + ([self.fallback] if self.fallback and self.fallback != self.model_in_use else [])
        last_exc: Exception | None = None
        for model in models:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._client.aio.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                    if model != self.model_in_use:
                        logger.warning("Switched planner to fallback model %s", model)
                        self.model_in_use = model
                    return response
                except errors.APIError as exc:
                    last_exc = exc
                    if exc.code not in RETRYABLE_CODES:
                        raise LLMError(f"Gemini error {exc.code}: {exc.message}") from exc
                    if exc.code in (429, 503, 504) and attempt >= 1:
                        break  # quota or overloaded model: switch to the fallback model instead of waiting
                    delay = min(2**attempt + random.random(), 20)  # noqa: S311 (jitter, not crypto)
                    logger.warning("Gemini %s on %s, retrying in %.1fs", exc.code, model, delay)
                    await asyncio.sleep(delay)
                except (TimeoutError, OSError) as exc:
                    last_exc = exc
                    await asyncio.sleep(min(2**attempt, 10))
        raise LLMError(f"Gemini request failed after retries: {last_exc}")
