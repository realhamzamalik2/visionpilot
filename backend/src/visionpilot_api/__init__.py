"""VisionPilot agent backend: FastAPI + Gemini planner + VisionPilot MCP tools."""

__version__ = "0.1.0"


def main() -> None:
    """Console entry point (`uv run visionpilot-api`)."""
    import logging

    import uvicorn

    from visionpilot_api.config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    uvicorn.run(
        "visionpilot_api.app:create_app", factory=True, host=settings.api_host, port=settings.api_port, log_level="info"
    )
