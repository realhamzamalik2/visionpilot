"""VisionPilot MCP server: pretrained, CPU-friendly vision models exposed as MCP tools."""

__version__ = "0.1.0"


def main() -> None:
    """Console entry point (`uv run visionpilot-mcp`)."""
    from visionpilot_mcp.server import main as _main

    _main()
