# VisionPilot MCP server (CPU only). Build context: mcp_server/
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first so code changes do not reinstall PyTorch.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

ENV MCP_TRANSPORT=streamable-http MCP_HTTP_HOST=0.0.0.0 MCP_HTTP_PORT=8765 \
    MODEL_DIR=/models HF_HOME=/models/huggingface LOG_DIR=/app/logs
EXPOSE 8765
CMD ["uv", "run", "--no-sync", "visionpilot-mcp"]
