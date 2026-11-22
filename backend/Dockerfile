# VisionPilot agent API. Build context: backend/
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

ENV API_HOST=0.0.0.0 API_PORT=8000 MCP_URL=http://mcp:8765/mcp DATA_DIR=/data SAMPLES_DIR=/samples
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "visionpilot-api"]
