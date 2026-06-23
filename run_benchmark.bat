@echo off
cd /d "%~dp0"

echo ==============================================
echo  ABEL Ablation Benchmark Suite
echo ==============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found at .venv\
    echo Run run_abel.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

echo Launching benchmark GUI...
echo.
".venv\Scripts\python.exe" -m abel.benchmark 2>&1
echo.
echo Exit code: %ERRORLEVEL%
echo.
pause
