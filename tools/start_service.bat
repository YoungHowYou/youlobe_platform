@echo off
REM Labelyou backend service wrapper (Task Scheduler @startup as SYSTEM).
REM Runs uvicorn with no console window; logs appended under logs\.
cd /d D:\yolo26s_platform
set YOLO_AUTOINSTALL=False
if not exist "logs" mkdir "logs"
venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000 >> logs\backend.out.log 2>> logs\backend.err.log
