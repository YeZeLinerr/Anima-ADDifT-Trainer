@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PARENT_ROOT=%PROJECT_DIR%.."
set "PYTHON=%PROJECT_DIR%venv\Scripts\python.exe"

if not exist "%PYTHON%" set "PYTHON=%PARENT_ROOT%\venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] Python venv was not found.
    echo Checked:
    echo   %PROJECT_DIR%venv\Scripts\python.exe
    echo   %PARENT_ROOT%\venv\Scripts\python.exe
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "TOKENIZERS_PARALLELISM=false"

cd /d "%PROJECT_DIR%"
echo Starting Anima ADDifT WebUI at http://127.0.0.1:3001
echo Python: %PYTHON%
echo.
"%PYTHON%" "%PROJECT_DIR%tools\anima_addift_webui.py" --host 127.0.0.1 --port 3001

echo.
echo WebUI exited.
pause
