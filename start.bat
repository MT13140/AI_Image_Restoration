@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"

echo ==================================================================
echo   AI Image Restoration  -  Launcher
echo   (The web UI itself is in Chinese)
echo ==================================================================

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "NEED_INSTALL=0"

if not exist "%VENV_PY%" (
    echo [1/3] No virtual environment found, creating .venv ...
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python not found. Please install Python 3.10 / 3.11 first.
        pause
        exit /b 1
    )
    python -m venv ".venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
    set "NEED_INSTALL=1"
) else (
    echo [1/3] Virtual environment found.
)

"%VENV_PY%" -c "import gradio, numpy, cv2, PIL, torch, rapidocr_onnxruntime, simple_lama_inpainting" >nul 2>nul
if errorlevel 1 set "NEED_INSTALL=1"

if "%NEED_INSTALL%"=="1" (
    echo [2/3] Installing / repairing dependencies, please wait ...
    call "%~dp0install.bat" /nopause
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. See the log above.
        pause
        exit /b 1
    )
) else (
    echo [2/3] Dependency check passed.
)

echo [3/3] Starting the web UI ...
echo.
"%VENV_PY%" "%~dp0app.py" %*

echo.
echo Server stopped.
pause
