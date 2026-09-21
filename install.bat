@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ==================================================================
echo   AI Image Restoration  -  Install dependencies
echo ==================================================================

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PIP_MIRROR=-i https://pypi.tuna.tsinghua.edu.cn/simple"

if not exist "%VENV_PY%" (
    echo [1/5] Creating virtual environment .venv ...
    python -m venv ".venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create the venv. Please install Python 3.10 / 3.11.
        if /i not "%~1"=="/nopause" pause
        exit /b 1
    )
) else (
    echo [1/5] Virtual environment already exists.
)

echo [2/5] Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip %PIP_MIRROR% --disable-pip-version-check

echo [3/5] Installing base dependencies ...
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt" %PIP_MIRROR% --disable-pip-version-check
if errorlevel 1 (
    echo [ERROR] Base dependency installation failed.
    if /i not "%~1"=="/nopause" pause
    exit /b 1
)

echo [4/5] Pinning LaMa wrapper and Pillow versions ...
"%VENV_PY%" -m pip install --no-deps "simple-lama-inpainting==0.1.2" "Pillow==10.4.0" %PIP_MIRROR% --disable-pip-version-check

echo [5/5] Installing PyTorch (CUDA 12.1 build first, CPU build as fallback) ...
"%VENV_PY%" -m pip install "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" --index-url https://download.pytorch.org/whl/cu121 --disable-pip-version-check
if errorlevel 1 (
    echo [INFO] CUDA build failed, trying the CPU build ...
    "%VENV_PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check
    if errorlevel 1 (
        echo [ERROR] PyTorch installation failed. Please check your network.
        if /i not "%~1"=="/nopause" pause
        exit /b 1
    )
)

echo.
echo Dependencies installed. Run start.bat to launch the app.
if /i not "%~1"=="/nopause" pause
endlocal
exit /b 0
