@echo off
TITLE SignBridge - Launcher
echo ===================================================
echo  Starting SignBridge (Indian Sign Language App)
echo ===================================================

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH. Please install Python 3.9 - 3.11.
    pause
    exit /b
)

echo Checking dependencies...
python -m pip install -r requirements.txt --quiet

echo Launching application...
python app.py
pause
