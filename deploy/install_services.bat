@echo off
chcp 65001 >nul
REM ============================================================
REM  Register backend as a Windows service via NSSM.
REM  Frontend is served by the backend (web\dist), so NO separate
REM  frontend service is needed. After this, open the browser at
REM  http://localhost:8000  (admin / admin123).
REM
REM  ⚠ 备选方案：需要额外下载 nssm.exe 放到 deploy\ 目录。
REM    推荐直接用同目录下的 install_service.bat（基于 pywin32，
REM    零外部依赖，也是官方推荐路径）。
REM
REM  本平台只依赖本地硬盘（SQLite 文件库），不再需要 MySQL/Redis/Celery，
REM  后端单进程内即可完成训练任务调度。
REM
REM  RUN AS ADMINISTRATOR, one time, on the target machine.
REM  Prereqs already done: Python venv ready, deps installed,
REM  web\dist built.
REM ============================================================

REM  App dir = parent of this deploy\ folder (auto-detected; folder name can be anything, e.g. Labelyou_AI)
for %%I in ("%~dp0..") do set "APP_DIR=%%~fI"
set PY=%APP_DIR%\venv\Scripts\python.exe

REM  nssm.exe: put it next to this script, or set full path / add to PATH.
set NSSM=%~dp0nssm.exe
if not exist "%NSSM%" set NSSM=nssm.exe

if not exist "%APP_DIR%\logs" mkdir "%APP_DIR%\logs"

echo === Registering LabelyouBackend (uvicorn: API + web UI + 内置训练线程) ===
REM 注意：AppParameters 必须拆成独立参数传，整体加引号会被当成单个参数导致启动失败
"%NSSM%" install LabelyouBackend "%PY%" -m uvicorn server.main:app --host 0.0.0.0 --port 8000
"%NSSM%" set LabelyouBackend AppDirectory "%APP_DIR%"
"%NSSM%" set LabelyouBackend AppEnvironmentExtra YOLO_AUTOINSTALL=False PYTHONPATH=%APP_DIR%
"%NSSM%" set LabelyouBackend AppStdout "%APP_DIR%\logs\backend.out.log"
"%NSSM%" set LabelyouBackend AppStderr "%APP_DIR%\logs\backend.err.log"
"%NSSM%" set LabelyouBackend AppExit Default Restart
"%NSSM%" set LabelyouBackend AppRestartDelay 5000
"%NSSM%" set LabelyouBackend Start SERVICE_AUTO_START

echo === Starting service ===
net start LabelyouBackend
LabelyouBackend
echo.
echo To remove later (as admin):
echo   net stop LabelyouBackend ^& "%NSSM%" remove LabelyouBedit VP_backend
echo.
echo To remove later (as admin):
echo   net stop VP_backend ^& "%NSSM%" remove VP_backend confirm
pause
