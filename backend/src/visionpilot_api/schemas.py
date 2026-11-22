"""API and agent schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

MediaKind = Literal["image", "video"]


class MediaInfo(BaseModel):
    id: str = Field(description="'upload:<uuid>.<ext>' or 'sample:<scenario>/<file>'")
    kind: MediaKind
    filename: str
    url: str = Field(description="Where the frontend can load the file.")
    width: int | None = None
    height: int | None = None
    size_bytes: int


class SampleInfo(MediaInfo):
    scenario: str
    title: str
    suggested_questions: list[str] = Field(default_factory=list)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class ChatRequest(BaseModel):
    media_id: str
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class Evidence(BaseModel):
    """Links a claim in the answer to tool output, so the UI can highlight boxes on hover."""

    step: int = Field(description="Step number of the tool call that supports the claim.")
    item_indices: list[int] = Field(default_factory=list, description="Indices into that tool's result list.")
    label: str = Field(description="Short phrase from the answer that this evidence supports.")


class FinalAnswer(BaseModel):
    answer: str = Field(description="Markdown answer for the user.")
    evidence: list[Evidence] = Field(default_factory=list)


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCallResult(BaseModel):
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class TraceEvent(BaseModel):
    """One server-sent event. `type` tells the frontend which card to render."""

    type: Literal["run_started", "thinking", "tool_call", "tool_result", "answer", "error", "done"]
    run_id: str
    step: int | None = None
    call_id: str | None = None
    tool: str | None = None
    args: dict[str, Any] | None = None
    ok: bool | None = None
    summary: str | None = None
    latency_ms: float | None = None
    data: dict[str, Any] | None = None
    message: str | None = None
    answer: FinalAnswer | None = None
    model: str | None = None
