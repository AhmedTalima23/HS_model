@echo off
echo.
echo ========================================
echo   PRISMA Classification Streamlit App
echo ========================================
echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Python found!
echo.
echo Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing Streamlit...
    pip install streamlit
)

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Installing PyTorch...
    pip install torch torchvision
)

echo.
echo Launching Streamlit application...
echo The app will open in your default web browser.
echo If it doesn't open automatically, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

python -m streamlit run streamlit_app.py

echo.
echo Application stopped.
pause
