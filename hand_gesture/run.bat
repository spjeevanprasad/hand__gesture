@echo off
cd /d "%~dp0"
title Hand Gesture Recognition
echo ======================================================================
echo   Launching Hand Gesture Recognition (21 Skeleton Joints)
echo ======================================================================

set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" main.py %*
) else (
    python main.py %*
)

echo.
echo Application finished.
pause
