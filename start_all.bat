@echo off
chcp 65001 >nul
echo ========================================
echo   YOLO26s-Seg Platform Starting...
echo ========================================

echo [1/2] Starting Backend (uvicorn, 内置训练线程)...
start "" /min "%~dp0start_backend.bat"
timeout /t 5 /nobreak >nul

echo [2/2] Starting Frontend...
start "" /min "%~dp0start_frontend.bat"
timeout /t 3 /nobreak >nul

echo ========================================
echo   All services started!
echo   URL: http://localhost:5174
echo   Default account: admin / admin123
echo ========================================
pause
