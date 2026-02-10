@echo off
cls
echo ================================================
echo    Neural Forex Trader - Windows Application
echo ================================================
echo.
echo Starting trading application...
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run the GUI application
python neural_trader_gui.py

if errorlevel 1 (
    echo.
    echo Error: Application crashed or MT5 not connected
    echo Make sure MetaTrader 5 is running and logged in
    pause
)
