@echo off
chcp 65001 >nul
REM ============================================================
REM  卸载 Windows 服务 LabelyouBackend
REM  【必须以管理员身份运行】
REM
REM  只删除服务注册，不会删除 storage\ 里的数据和代码。
REM ============================================================

net session >nul 2>&1
if errorlevel 1 (
    echo.
    echo [错误] 需要管理员权限。请右键本文件，选择"以管理员身份运行"。
    echo.
    pause
    exit /b 1
)

for %%I in ("%~dp0..") do set "APP_DIR=%%~fI"
set "PY=%APP_DIR%\venv\Scripts\python.exe"
set "SVC=LabelyouBackend"

echo ============================================================
echo   卸载服务 %SVC%
echo ============================================================
echo.

sc query %SVC% >nul 2>&1
if errorlevel 1 (
    echo 服务 %SVC% 未安装，无需卸载。
    echo.
    pause
    exit /b 0
)

echo [1/4] 停止服务 ...
net stop %SVC%

echo [2/4] 删除服务注册 ...
if exist "%PY%" (
    "%PY%" "%APP_DIR%\tools\win_service.py" remove
) else (
    sc delete %SVC%
)

echo [3/4] 移除防火墙放行规则 ...
if exist "%PY%" (
    "%PY%" "%APP_DIR%\tools\win_service.py" close-firewall
) else (
    netsh advfirewall firewall delete rule name="LabelyouBackend-8000"
)

echo [4/4] 确认结果 ...
sc query %SVC% >nul 2>&1
if errorlevel 1 (
    echo.
    echo 已卸载。数据仍保留在 %APP_DIR%\storage
    echo 如需恢复手动启动：双击 start_backend.bat
) else (
    echo.
    echo [警告] 服务似乎仍然存在，请手动执行： sc delete %SVC%
)
echo.
pause
