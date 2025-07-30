@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
call venv\Scripts\python.exe src\webcam_app.py
pause
