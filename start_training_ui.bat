@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Install Python or create a project venv, then try again.
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
set "TOKENIZERS_PARALLELISM=false"

echo Starting Anima ADDifT WebUI at http://127.0.0.1:3001
"%PYTHON%" "tools\anima_addift_webui.py" --host 127.0.0.1 --port 3001

echo.
echo WebUI exited.
pause
