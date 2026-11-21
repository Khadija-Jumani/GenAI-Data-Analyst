@echo off
echo ========================================================
echo   Starting GenAI Data Analyst Backend Server...
echo   Open in browser: http://localhost:8000/docs
echo ========================================================

if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe run.py
) else (
    python run.py
)
pause
