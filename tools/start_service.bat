@echo off
REM ============================================================
REM  后台运行后端（方案 B：不带 SCM，用于计划任务或临时后台实例）
REM
REM  标准部署请用 deploy\install_service.bat 注册成真正的 Windows 服务
REM  （开机自启 + 崩溃自动重启 + services.msc 可管理）。
REM  本脚本适用于：不想引入 pywin32，或想临时在后台再跑一个实例。
REM
REM  配套的计划任务（管理员 CMD 执行，开机以 SYSTEM 身份启动）：
REM    schtasks /Create /TN LabelyouBackend /SC ONSTART /RU SYSTEM /RL HIGHEST /F ^
REM             /TR "\"%~dp0start_service.bat\""
REM ============================================================
cd /d "%~dp0.."
set YOLO_AUTOINSTALL=False
if not exist "logs" mkdir "logs"
venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000 >> logs\backend.out.log 2>> logs\backend.err.log
