"""Errors that are safe to show to the calling model or user."""

from __future__ import annotations

from mcp.server.mcpserver.exceptions import ToolError


class VisionToolError(ToolError):
    """A tool failed for a reason the caller can act on (bad input, missing model, etc.).

    Subclassing MCP's ToolError makes the SDK pass this message to the client as an is_error result,
    instead of replacing it with a generic "Error executing tool" message.
    """


class ImageLoadError(VisionToolError):
    """The image could not be read from the given path, URL or data URI."""
