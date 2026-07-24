@echo off
echo Setting up AeroSentinel...

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Setup complete. 
echo To run the dashboard, use:
echo call venv\Scripts\activate
echo streamlit run src/dashboard/app.py
pause
