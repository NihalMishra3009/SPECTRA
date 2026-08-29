#Requires -Version 5.1
# SPECTRA Launcher - starts database (seeded), backend API, and dashboard.
# Usage:  .\run.ps1     (run from the repo root)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root "backend\.venv\Scripts\python.exe"

Write-Host "== SPECTRA launcher ==" -ForegroundColor Cyan

# 1) Python env
if (-not (Test-Path $py)) {
    Write-Host "[x] venv missing at $py`n    Setup first:" -ForegroundColor Red
    Write-Host "    cd backend; python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

# 2) Database - auto-init schema and seed representative runs when empty
$runs = 0
& $py -c "import sys; sys.path.insert(0, r'$root\backend'); from app import db; print(db.stats()['total_runs'])" |
    ForEach-Object { $runs = [int]$_ }
if (-not ($runs -gt 0)) {
    Write-Host "[i] database empty -> seeding sample runs..." -ForegroundColor Yellow
    Push-Location (Join-Path $root "backend")
    & $py -m app.seed_db
    Pop-Location
    Write-Host "[i] seed complete" -ForegroundColor Green
}
else {
    Write-Host "[i] database ready -> database/spectra.db ($runs recorded runs)" -ForegroundColor Green
}

# 3) Backend API  http://127.0.0.1:8000
$be = Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory (Join-Path $root "backend") -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 3
Write-Host "[i] backend  -> http://127.0.0.1:8000  (pid $($be.Id))" -ForegroundColor Green

# 4) Dashboard  http://localhost:5173 (runs in this terminal; Ctrl+C to stop)
Write-Host "[i] dashboard -> http://localhost:5173  (Ctrl+C stops launcher)" -ForegroundColor Green
Push-Location (Join-Path $root "frontend")
if (-not (Test-Path "node_modules")) { npm install }
npm run dev