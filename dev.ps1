# Starts VisionPilot locally in three new PowerShell windows: MCP server, agent API, web app.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File .\dev.ps1
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\mcp_server'; uv run visionpilot-mcp --transport streamable-http"
Start-Sleep -Seconds 3
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; uv run visionpilot-api"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"
Write-Host "Web app:    http://localhost:3000"
Write-Host "API docs:   http://127.0.0.1:8000/docs"
Write-Host "MCP server: http://127.0.0.1:8765/mcp"
