@echo off
title YOLO Frontend
cd /d "%~dp0web"
npx vite --host 0.0.0.0 --port 5174
pause
