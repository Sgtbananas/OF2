@echo off
REM ============================
REM  OnlyFunds Quick Start Script
REM ============================
REM This batch file launches your Streamlit trading app.
REM Double-click to run from Windows Explorer.

REM Change to the script's directory so relative paths work
cd /d "%~dp0"

REM (Optional) Install requirements if not already present
pip show streamlit >nul 2>nul
if errorlevel 1 (
    echo Installing required Python packages...
    pip install -r requirements.txt
)

REM Launch the Streamlit app
streamlit run main.py

pause