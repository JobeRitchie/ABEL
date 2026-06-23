@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ABEL one-click launcher for Windows.
REM - Creates .venv with Python 3.12 (fallback to 3.11) if missing
REM - Installs/updates project in editable mode
REM - Launches the app

cd /d "%~dp0"

if not exist "logs" mkdir "logs"
set "RUN_LOG=logs\launcher_last.log"
echo ==== ABEL launcher run: %DATE% %TIME% ==== > "%RUN_LOG%"

echo ==============================================
echo ABEL Launcher
echo Project: %CD%
echo ==============================================

set "PY_EXE="
set "CREATOR_PY="
set "NEED_RECREATE=0"

REM --- Find a Python 3.10+ executable to create the venv with ---
REM Prefer standalone python.org installs (avoids Conda DLL baggage in PATH).
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "CREATOR_PY=%LocalAppData%\Programs\Python\Python312\python.exe"
if "%CREATOR_PY%"=="" if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "CREATOR_PY=%LocalAppData%\Programs\Python\Python311\python.exe"
if "%CREATOR_PY%"=="" if exist "%LocalAppData%\Programs\Python\Python310\python.exe" set "CREATOR_PY=%LocalAppData%\Programs\Python\Python310\python.exe"

REM Fall back to py.exe launcher (python.org), then bare python (may be Anaconda).
if "%CREATOR_PY%"=="" (
  where py >nul 2>nul
  if not errorlevel 1 set "CREATOR_PY=py"
)
if "%CREATOR_PY%"=="" (
  where python >nul 2>nul
  if not errorlevel 1 set "CREATOR_PY=python"
)

if "%CREATOR_PY%"=="" (
  echo [ERROR] No Python interpreter found.
  echo Install Python 3.10+ from python.org and rerun this file.
  pause
  exit /b 1
)

REM --- Create or validate venv ---
set "PY_EXE=.venv\Scripts\python.exe"

if not exist "%PY_EXE%" (
  echo [INFO] No virtual environment found. Creating .venv...
  set "NEED_RECREATE=1"
)

if "%NEED_RECREATE%"=="0" (
  "%PY_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo [WARN] Existing .venv uses Python below 3.10. Recreating...
    set "NEED_RECREATE=1"
  )
)

if "%NEED_RECREATE%"=="1" (
  if exist ".venv" rmdir /s /q ".venv"
  echo [INFO] Running: %CREATOR_PY% -m venv .venv
  "%CREATOR_PY%" -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create virtual environment.
    pause
    exit /b 1
  )
)

if not exist "%PY_EXE%" (
  echo [ERROR] venv creation silently failed - python.exe missing at %PY_EXE%
  pause
  exit /b 1
)

:NON_CONDA_OK

"%PY_EXE%" --version >> "%RUN_LOG%" 2>&1

REM Reduce Qt/Conda DLL conflicts by clearing variables that can override plugin/runtime discovery.
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "PYTHONPATH="
set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "CONDA_SHLVL="

REM --- Decide whether a (re)install is actually needed ---
REM ABEL is installed editable (-e), so source edits are live without
REM reinstalling. We only need to install on first run, when the venv was
REM recreated, or when pyproject.toml changed (deps/version/entry points).
REM A stamp file under .venv records the last successful install; if it is
REM at least as new as pyproject.toml and abel still imports, we skip.
set "INSTALL_STAMP=.venv\.abel_install_stamp"
set "NEED_INSTALL=1"
"%PY_EXE%" -m abel._install_check >nul 2>nul
if not errorlevel 1 set "NEED_INSTALL=0"

if "%NEED_INSTALL%"=="0" goto SKIP_INSTALL

REM Pip/setuptools/wheel only need upgrading on a freshly created venv; on a
REM routine reinstall (e.g. a version bump) they are already current.
if "%NEED_RECREATE%"=="1" (
  echo [INFO] Preparing pip tooling...
  "%PY_EXE%" -m pip install --quiet --upgrade pip setuptools wheel >> "%RUN_LOG%" 2>&1
  if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip tooling. See %RUN_LOG%.
    pause
    exit /b 1
  )
)

echo [INFO] Installing/updating ABEL ^(first run may take a few minutes^)...
"%PY_EXE%" -m pip install --quiet -e . >> "%RUN_LOG%" 2>&1
if errorlevel 1 (
  echo [ERROR] Failed to install project dependencies. See %RUN_LOG%.
  pause
  exit /b 1
)

REM Record a successful install so subsequent launches can skip it.
type nul > "%INSTALL_STAMP%"
goto INSTALL_DONE

:SKIP_INSTALL
echo [INFO] ABEL is up to date.

:INSTALL_DONE

REM For runtime, force a clean PATH so Qt DLL resolution does not pick up Anaconda/system Qt binaries.
set "PATH=%CD%\.venv\Scripts;%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0\"

"%PY_EXE%" -c "from PySide6 import QtWidgets" >nul 2>> "%RUN_LOG%"
if errorlevel 1 (
  echo [WARN] PySide6 self-test failed. Trying stable fallback PySide6==6.7.3...
  "%PY_EXE%" -m pip install --force-reinstall "PySide6==6.7.3" "PySide6_Addons==6.7.3" "PySide6_Essentials==6.7.3" "shiboken6==6.7.3"
  if errorlevel 1 (
    echo [ERROR] Failed to install PySide6 fallback.
    pause
    exit /b 1
  )

  "%PY_EXE%" -c "from PySide6 import QtWidgets" >nul 2>> "%RUN_LOG%"
  if errorlevel 1 (
    echo.
    echo [ERROR] PySide6 still failed to import.
    echo [HINT] Install or repair Microsoft Visual C++ Redistributable 2015-2022 x64.
    echo [HINT] Get it from: https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo [HINT] After installing the redistributable, reboot and run this launcher again.
    pause
    exit /b 1
  )
)

echo [INFO] Launching ABEL...
"%PY_EXE%" -m abel.main 2>> "%RUN_LOG%"
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
  echo [WARN] ABEL exited with code %APP_EXIT%.
  echo [WARN] ABEL exited with code %APP_EXIT%. >> "%RUN_LOG%"
  echo.
  echo === Last log entries ===
  type "%RUN_LOG%"
  echo ========================
  echo [INFO] Full log: %CD%\%RUN_LOG%
  pause
)

if "%APP_EXIT%"=="0" (
  echo [INFO] ABEL exited normally.
  echo [INFO] Launcher log: %RUN_LOG%
  echo [INFO] Press any key to close this window.
  pause >nul
)

pause
exit /b %APP_EXIT%
