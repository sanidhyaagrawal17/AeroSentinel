@echo off
echo Starting AeroSentinel Dashboard...

if not exist venv (
    echo Error: Virtual environment not found. Please run setup.bat first.
    pause
    exit /b
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Launching Streamlit application...
streamlit run src/dashboard/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
