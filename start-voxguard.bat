@echo off
echo Starting VoxGuard - 3 services...

start "VoxGuard - FastAPI (ML)" cmd /k "cd /d C:\VoxGuard && c:\VoxGuard\ml\.venv\Scripts\Activate.ps1 && uvicorn main:app --reload --port 8000"

timeout /t 3

start "VoxGuard - Backend (Node)" cmd /k "cd /d C:\VoxGuard && npm run dev"

timeout /t 3

start "VoxGuard - Frontend" cmd /k "cd /d C:\VoxGuard\frontend && npm run dev"

echo All 3 services starting in separate windows.
echo Wait ~15 seconds, then open http://localhost:3000
pause
