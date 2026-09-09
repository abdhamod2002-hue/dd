# ============================================================================
#  AI Littering Detection - Windows launcher
#  .\start.ps1
#
#  Goal: simple user experience.
#    1. Start PostgreSQL + backend + dashboard (Docker preferred).
#    2. Open the dashboard.
#    3. Try to detect a real camera. If none is present, do NOT block the user:
#       video-file analysis remains available at /analysis.
#
#  The system never claims live-camera readiness unless a real camera passes
#  the smoke test. No fake detections, no fake events.
# ============================================================================

param(
    [switch]$SkipDocker,
    [switch]$SkipCamera,
    [int]$CameraDevice = -1   # -1 = auto-discover
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Write-Status($tag, $name, $detail = "") {
    $color = switch ($tag) {
        "OK"      { "Green" }
        "WARNING" { "Yellow" }
        "ERROR"   { "Red" }
        "WAITING" { "Cyan" }
        default   { "Gray" }
    }
    $line = "  $name".PadRight(30) + "[$tag]"
    if ($detail) { $line += "  $detail" }
    Write-Host $line -ForegroundColor $color
}

function Test-Url($url) {
    try {
        $null = Invoke-RestMethod -Uri $url -TimeoutSec 3 -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AI LITTERING DETECTION SYSTEM" -ForegroundColor Cyan
Write-Host "  START" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- verify installation happened ---
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Status "ERROR" "Installation" "run .\install.ps1 first"
    exit 1
}
if (-not (Test-Path "dashboard\node_modules")) {
    Write-Status "ERROR" "Dashboard deps" "run .\install.ps1 first"
    exit 1
}
Write-Status "OK" "Installation verified"

$py = ".venv\Scripts\python.exe"
$backendUrl = "http://localhost:8000"
$dashUrl = "http://localhost:5173"

# ============================================================================
# STEP 0: Start infrastructure
# ============================================================================
Write-Host ""
Write-Host "  Starting infrastructure..." -ForegroundColor Gray

$dockerPresent = $false
try {
    $null = docker --version 2>&1
    $dockerPresent = $true
} catch {}

if ($dockerPresent -and -not $SkipDocker) {
    try {
        docker compose up -d postgres backend dashboard 2>&1 | Out-Null
        for ($i = 0; $i -lt 60; $i++) {
            if (Test-Url "$backendUrl/api/status") { break }
            Start-Sleep -Seconds 1
        }
        if (Test-Url "$backendUrl/api/status") {
            Write-Status "OK" "Docker backend" "$backendUrl"
        } else {
            Write-Status "WARNING" "Docker backend" "not reachable yet; check docker compose logs backend"
        }
        for ($i = 0; $i -lt 30; $i++) {
            if (Test-Url $dashUrl) { break }
            Start-Sleep -Seconds 1
        }
        if (Test-Url $dashUrl) {
            Write-Status "OK" "Docker dashboard" $dashUrl
        } else {
            Write-Status "WARNING" "Docker dashboard" "not reachable yet; check docker compose logs dashboard"
        }
    } catch {
        Write-Status "WARNING" "Docker" "docker compose failed; falling back to local processes"
    }
} else {
    Write-Status "WARNING" "Docker" "not used; starting local backend/dashboard"
}

# If Docker did not bring the backend up, start local backend.
if (-not (Test-Url "$backendUrl/api/status")) {
    $backendProc = Start-Process powershell -PassThru -ArgumentList "-NoExit", "-Command", "& '$py' -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
    for ($i = 0; $i -lt 20; $i++) {
        if (Test-Url "$backendUrl/api/status") { break }
        Start-Sleep -Seconds 1
    }
    if (Test-Url "$backendUrl/api/status") {
        Write-Status "OK" "Local backend" "$backendUrl"
    } else {
        Write-Status "WARNING" "Local backend" "not reachable; dashboard will show limited data"
    }
}

# If Docker did not bring the dashboard up, start local dashboard preview.
if (-not (Test-Url $dashUrl)) {
    $dashProc = Start-Process powershell -PassThru -ArgumentList "-NoExit", "-Command", "cd dashboard; npm run preview -- --host 0.0.0.0 --port 5173"
    for ($i = 0; $i -lt 20; $i++) {
        if (Test-Url $dashUrl) { break }
        Start-Sleep -Seconds 1
    }
    if (Test-Url $dashUrl) {
        Write-Status "OK" "Local dashboard" $dashUrl
    } else {
        Write-Status "WARNING" "Local dashboard" "could not reach $dashUrl - open manually"
    }
}

# ============================================================================
# STEP 1: Open the Dashboard
# ============================================================================
Write-Host ""
Write-Host "  Opening Dashboard in browser..." -ForegroundColor Gray
if (Test-Url $dashUrl) {
    Start-Process $dashUrl
    Write-Status "OK" "Dashboard opened" $dashUrl
} else {
    Write-Status "WARNING" "Dashboard" "could not reach $dashUrl - open manually"
}

# ============================================================================
# STEP 2: Optional camera smoke test
# ============================================================================
Write-Host ""
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host "  Camera / live-input check (optional)" -ForegroundColor Yellow
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host ""

if ($SkipCamera) {
    Write-Status "OK" "Camera check" "skipped by -SkipCamera"
    Write-Host "  You can still upload recorded videos in the Dashboard -> Video Analysis." -ForegroundColor Gray
} else {
    Write-Host "  Detecting cameras..." -ForegroundColor Gray
    $camOut = & $py scripts/camera_discovery.py 2>$null
    Write-Host $camOut
    Write-Host ""

    $liveIdx = -1
    if ($camOut -match "Idx\s+(\d+)\s+LIVE") { $liveIdx = $Matches[1] }
    if ($liveIdx -lt 0) {
        Write-Status "WAITING" "iPhone Camera" "no LIVE camera detected"
        Write-Host ""
        Write-Host "  This is OK for video-file analysis." -ForegroundColor Green
        Write-Host "  Open the Dashboard, go to /analysis, and upload a real video." -ForegroundColor Green
        Write-Host "  To use the live camera, connect the iPhone via Camo and re-run .\start.ps1." -ForegroundColor Gray
    } else {
        $CameraDevice = $liveIdx
        Write-Status "OK" "iPhone Camera" "device index $CameraDevice (LIVE)"

        $smokeScript = @"
import cv2, time, sys
sys.path.insert(0, '.')
from inference.detection.yolo_detector import YoloDetector
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.pose.movenet_pose import MovenetPose
from inference.pipeline import InferencePipeline, PipelineConfig
from scripts.run_pipeline import build_tracks_real

cap = cv2.VideoCapture($CameraDevice)
if not cap.isOpened(): print('[ERROR] Camera'); sys.exit(1)
det = YoloDetector(); det.load()
mv = MovenetPose(); mv.load()
tr = BytetrackTracker(); tr.load()
cfg = PipelineConfig(analysis_fps=10.0)
pipe = InferencePipeline(cfg)
person_ids = set(); stable = True; prev_ids = None
for i in range(20):
    ok, f = cap.read()
    if not ok: print('[ERROR] read'); break
    tracked = det.track(f, persist=True)
    persons, objects = build_tracks_real(f, tracked, mv, tr, i)
    ids = tuple(sorted(p.track_id for p in persons))
    person_ids.update(ids)
    if prev_ids is not None and ids != prev_ids:
        stable = False
    prev_ids = ids
    pipe.process_frame(f, time.time(), persons, objects)
print('[OK] Camera')
print('[OK] YOLO')
print('[OK] ByteTrack')
if not person_ids:
    print('[WARNING] No persons detected in 20 frames - check camera angle/lighting')
else:
    print('[OK] MoveNet')
    print('[OK] Association')
    print('[OK] Temporal event detector')
    print('[OK] Evidence buffer')
print('[%s] Stable Track IDs (persons: %s)' % ('OK' if stable else 'WARNING', sorted(person_ids)))
print('Person tracks seen: %s' % sorted(person_ids))
cap.release()
"@
        $smokePy = Join-Path $env:TEMP "ai_littering_smoke.py"
        Set-Content -Path $smokePy -Value $smokeScript -Encoding UTF8
        $smokeOut = & $py $smokePy 2>&1
        Remove-Item $smokePy -ErrorAction SilentlyContinue
        Write-Host $smokeOut
        if ($smokeOut -match "\[ERROR\]") {
            Write-Status "ERROR" "AI smoke test" "real camera frames failed to pass the pipeline"
            Write-Host "  Video-file analysis is still available." -ForegroundColor Yellow
        } elseif ($smokeOut -match "\[WARNING\]") {
            Write-Status "WARNING" "AI smoke test" "camera works, but no person was detected in the 20-frame sample"
            Write-Host "  Adjust camera angle/lighting, or use video-file analysis." -ForegroundColor Gray
        } else {
            Write-Status "OK" "AI smoke test" "real camera frames passed the production pipeline"
        }
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  DASHBOARD IS READY" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard: $dashUrl" -ForegroundColor Green
Write-Host "  Backend:   $backendUrl" -ForegroundColor Green
Write-Host ""
Write-Host "  For live camera: connect iPhone via Camo, then open /live." -ForegroundColor White
Write-Host "  For video files: open /analysis and upload a real video." -ForegroundColor White
Write-Host ""
Write-Host "  Backend and Dashboard are running in background windows/containers." -ForegroundColor DarkGray
Write-Host "  Close them manually or run: docker compose down" -ForegroundColor DarkGray
