# VisionPilot launcher: runs the MCP server, agent API and web app in ONE window.
#
#   powershell -ExecutionPolicy Bypass -File .\start.ps1
#   powershell -ExecutionPolicy Bypass -File .\start.ps1 -NoBrowser
#
# Every log line is prefixed with [mcp], [api] or [web]. The script waits until all
# three parts are ready, opens http://localhost:3000, and stops everything when you
# press Ctrl+C (or Q). Full logs are kept in logs\run\.

param(
    [switch]$NoBrowser,
    [int]$ReadyTimeoutSeconds = 240
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $root 'logs\run'
New-Item -ItemType Directory -Force $logDir | Out-Null

$WebUrl = 'http://localhost:3000'
$HealthUrl = 'http://127.0.0.1:8000/api/health'

function Write-Tag([string]$Tag, [string]$Color, [string]$Text) {
    Write-Host ('[{0}] ' -f $Tag) -ForegroundColor $Color -NoNewline
    Write-Host $Text
}

function Test-Port([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $attempt = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $attempt.AsyncWaitHandle.WaitOne(300)) { return $false }
        $client.EndConnect($attempt)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Get-Health {
    try {
        return Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
    } catch {
        return $null
    }
}

function Stop-Tree([System.Diagnostics.Process]$Process) {
    if ($null -eq $Process) { return }
    try {
        if (-not $Process.HasExited) {
            cmd /c "taskkill /PID $($Process.Id) /T /F >nul 2>&1" | Out-Null
        }
    } catch { }
}

# ---- Pre-flight checks -------------------------------------------------------
$uv = Get-Command uv -ErrorAction SilentlyContinue
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $uv) { Write-Host 'uv was not found. Install it first (see README).' -ForegroundColor Red; exit 1 }
if (-not $npm) { Write-Host 'npm was not found. Install Node.js first (see README).' -ForegroundColor Red; exit 1 }
if (-not (Test-Path (Join-Path $root '.env'))) {
    Write-Host 'No .env file found. Copy .env.example to .env and add your GEMINI_API_KEY.' -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $root 'frontend\node_modules'))) {
    Write-Host 'frontend\node_modules is missing. Run: cd frontend; npm install' -ForegroundColor Red
    exit 1
}

$busy = @()
foreach ($port in 8765, 8000, 3000) {
    if (Test-Port $port) { $busy += $port }
}
if ($busy.Count -gt 0) {
    Write-Host ('Port(s) already in use: {0}' -f ($busy -join ', ')) -ForegroundColor Red
    Write-Host 'VisionPilot (or another program) is probably already running. Close it first.'
    Write-Host 'To see what is using a port:  Get-NetTCPConnection -LocalPort 8765 | Select-Object OwningProcess'
    exit 1
}

# ---- Start the three parts (hidden, output redirected to log files) ------------
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:FORCE_COLOR = '0'

$services = @(
    @{ Tag = 'mcp'; Color = 'Cyan';    Dir = 'mcp_server'; Exe = $uv.Source;  Args = 'run visionpilot-mcp --transport streamable-http' },
    @{ Tag = 'api'; Color = 'Green';   Dir = 'backend';    Exe = $uv.Source;  Args = 'run visionpilot-api' },
    @{ Tag = 'web'; Color = 'Magenta'; Dir = 'frontend';   Exe = $npm.Source; Args = 'run dev' }
)

$previousCtrlC = [Console]::TreatControlCAsInput
$readers = @()

try {
    foreach ($s in $services) {
        $out = Join-Path $logDir ('{0}.out.log' -f $s.Tag)
        $err = Join-Path $logDir ('{0}.err.log' -f $s.Tag)
        $s.Process = Start-Process -FilePath $s.Exe -ArgumentList $s.Args `
            -WorkingDirectory (Join-Path $root $s.Dir) -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $out -RedirectStandardError $err
        $null = $s.Process.Handle   # keep a handle so the exit code stays readable
        foreach ($path in $out, $err) {
            $stream = [System.IO.File]::Open($path, 'OpenOrCreate', 'Read', 'ReadWrite')
            $readers += @{ Tag = $s.Tag; Color = $s.Color; Reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8) }
        }
        Write-Tag $s.Tag $s.Color ('started (PID {0})' -f $s.Process.Id)
    }

    Write-Host ''
    Write-Host 'Starting up. The MCP server needs about a minute to load the models...' -ForegroundColor Yellow
    Write-Host 'Press Ctrl+C or Q to stop everything.' -ForegroundColor Yellow
    Write-Host ''

    [Console]::TreatControlCAsInput = $true
    $started = Get-Date
    $lastCheck = [datetime]::MinValue
    $ready = $false
    $warnedSlow = $false

    while ($true) {
        # 1. Print any new log lines.
        foreach ($r in $readers) {
            while ($null -ne ($line = $r.Reader.ReadLine())) {
                if ($line.Trim().Length -gt 0) { Write-Tag $r.Tag $r.Color $line }
            }
        }

        # 2. Stop if any part crashed.
        $dead = $services | Where-Object { $_.Process.HasExited } | Select-Object -First 1
        if ($dead) {
            Write-Host ''
            Write-Host ('[{0}] stopped unexpectedly (exit code {1}). Stopping the others. Logs: {2}' -f `
                $dead.Tag, $dead.Process.ExitCode, $logDir) -ForegroundColor Red
            break
        }

        # 3. Check readiness once a second until everything is up.
        if (-not $ready -and ((Get-Date) - $lastCheck).TotalSeconds -ge 1) {
            $lastCheck = Get-Date
            $health = Get-Health
            $webUp = Test-Port 3000
            if ($health -and $health.mcp_connected -and $webUp) {
                $ready = $true
                $seconds = [int]((Get-Date) - $started).TotalSeconds
                Write-Host ''
                Write-Host ('VisionPilot is ready after {0} s.' -f $seconds) -ForegroundColor Green
                Write-Host ('  Web app:   {0}' -f $WebUrl)
                Write-Host '  API docs:  http://127.0.0.1:8000/docs'
                Write-Host ('  Model:     {0}  ({1} tools)' -f $health.model, @($health.tools).Count)
                if (-not $health.gemini_key_set) {
                    Write-Host '  Warning: GEMINI_API_KEY is empty in .env, so chat will not work.' -ForegroundColor Red
                }
                Write-Host '  Press Ctrl+C or Q to stop.' -ForegroundColor Yellow
                Write-Host ''
                if (-not $NoBrowser) { Start-Process $WebUrl }
            } elseif (-not $warnedSlow -and ((Get-Date) - $started).TotalSeconds -gt $ReadyTimeoutSeconds) {
                $warnedSlow = $true
                Write-Host ('Still not ready after {0} s. Check the log lines above, or the files in {1}' -f `
                    $ReadyTimeoutSeconds, $logDir) -ForegroundColor Red
            }
        }

        # 4. Ctrl+C or Q stops everything.
        if ([Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)
            $ctrlC = ($key.Key -eq 'C') -and (($key.Modifiers -band [ConsoleModifiers]::Control) -ne 0)
            if ($ctrlC -or $key.Key -eq 'Q') {
                Write-Host ''
                Write-Host 'Stopping VisionPilot...' -ForegroundColor Yellow
                break
            }
        }

        Start-Sleep -Milliseconds 200
    }
} finally {
    [Console]::TreatControlCAsInput = $previousCtrlC
    foreach ($s in $services) { Stop-Tree $s.Process }
    foreach ($r in $readers) { try { $r.Reader.Dispose() } catch { } }
    Write-Host 'All parts stopped.' -ForegroundColor Yellow
}
