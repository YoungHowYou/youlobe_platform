@echo off
chcp 65001 >nul
setlocal
REM ============================================================
REM  Allow inbound TCP 8000 so other machines on the LAN can
REM  open this platform (e.g. http://192.168.3.11:8000).
REM
REM  RUN AS ADMINISTRATOR  (right-click -> Run as administrator)
REM  Optional: pass --quiet to skip the final pause.
REM
REM  NOTE: this file is deliberately ASCII-only. Chinese text in a
REM  .bat file combined with "chcp 65001" makes cmd.exe mis-parse
REM  lines (they get executed as commands). All Chinese output
REM  comes from tools\win_service.py open-firewall instead.
REM ============================================================

for %%I in ("%~dp0..") do set "APP_DIR=%%~fI"
set "PY=%APP_DIR%\venv\Scripts\python.exe"

if not exist "%PY%" goto :novenv

"%PY%" "%APP_DIR%\tools\win_service.py" open-firewall
set "RC=%ERRORLEVEL%"
if /i not "%1"=="--quiet" pause
exit /b %RC%

:novenv
echo [ERROR] venv not found: %PY%
echo         Run first:  python tools\setup_env.py
if /i not "%1"=="--quiet" pause
exit /b 1
