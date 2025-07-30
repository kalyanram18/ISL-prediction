@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
call venv\Scripts\python.exe -m streamlit run src\app.py
pause
