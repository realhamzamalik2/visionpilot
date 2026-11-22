# Starts the VisionPilot MCP server (HTTP) and the agent API in two new PowerShell windows.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File .\dev.ps1
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\mcp_server'; uv run visionpilot-mcp --transport streamable-http"
Start-Sleep -Seconds 3
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; uv run visionpilot-api"
Write-Host "MCP server: http://127.0.0.1:8765/mcp"
Write-Host "API docs:   http://127.0.0.1:8000/docs   health: http://127.0.0.1:8000/api/health"
