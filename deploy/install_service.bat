@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
REM ============================================================
REM  把平台后端安装成 Windows 服务（开机自启、崩溃自动重启）
REM
REM  服务名 : LabelyouBackend
REM  访问   : http://localhost:8000     默认账号 admin / admin123
REM
REM  【必须以管理员身份运行】：右键本文件 → 以管理员身份运行
REM  前置条件：venv 已建好、依赖已装、web\dist 已构建
REM            （即已执行过 python tools\setup_env.py 与 npm run build）
REM ============================================================

REM ---- 0. 管理员检查 ----
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
echo   安装 Windows 服务 %SVC%
echo   项目目录：%APP_DIR%
echo ============================================================
echo.

if not exist "%PY%" (
    echo [错误] 没找到虚拟环境：%PY%
    echo        请先在项目根目录执行： python tools\setup_env.py
    echo.
    pause
    exit /b 1
)

REM ---- 1. 确保 pywin32 就位（服务宿主 pythonservice.exe 由它提供）----
"%PY%" -c "import win32serviceutil" >nul 2>&1
if errorlevel 1 (
    echo [1/5] 安装 pywin32 ...
    "%PY%" -m pip install pywin32
    if errorlevel 1 (
        echo [错误] pywin32 安装失败，请检查网络或 pip 源。
        pause
        exit /b 1
    )
) else (
    echo [1/5] pywin32 已就绪
)

REM ---- 2. 腾出 8000 端口（手动启动的实例还在跑的话服务起不来）----
echo [2/5] 检查并释放 8000 端口
"%PY%" "%APP_DIR%\tools\win_service.py" free-port
del "%APP_DIR%\backend.pid" >nul 2>&1

REM ---- 3. 注册服务（升级安装用 update）----
REM 注意：pywin32 注册失败时返回码仍是 0，只能靠 sc query 确认服务是否真的建好了
sc query %SVC% >nul 2>&1
if errorlevel 1 (
    echo [3/5] 注册服务 %SVC% ...
    "%PY%" "%APP_DIR%\tools\win_service.py" --startup auto install
) else (
    echo [3/5] 服务已存在，执行更新
    "%PY%" "%APP_DIR%\tools\win_service.py" --startup auto update
)
sc query %SVC% >nul 2>&1
if errorlevel 1 (
    echo.
    echo [错误] 服务注册失败（sc query 查不到 %SVC%）。
    echo        最常见原因：没有以管理员身份运行本脚本。
    echo        其他原因请看上面的报错信息。
    echo.
    pause
    exit /b 1
)

REM ---- 4. 崩溃自动重启 + 启动超时设置 ----
echo [4/5] 配置崩溃自动重启策略
sc failure %SVC% reset= 86400 actions= restart/5000/restart/10000/restart/60000 >nul
sc config %SVC% start= auto >nul
sc description %SVC% "Labelyou 工业缺陷检测平台后端：FastAPI + 前端托管 + 进程内训练线程（SQLite 本地存储，端口 8000）" >nul

REM ---- 放行防火墙：局域网内其它电脑才能打开本平台（本脚本已是管理员）----
call "%~dp0open_firewall.bat" --quiet

REM ---- 5. 启动并自检 ----
echo [5/5] 启动服务 ...
net start %SVC%
if errorlevel 1 (
    echo.
    echo [警告] 服务启动失败，下面是自动诊断结果：
    "%PY%" "%APP_DIR%\tools\win_service.py" diag
    echo.
    pause
    exit /b 1
)

echo.
echo 等待服务就绪 ...
"%PY%" "%APP_DIR%\tools\win_service.py" wait-health
REM errorlevel 必须紧跟被检查的命令，中间的 echo/set 会把它冲掉
if errorlevel 1 (set "READY=") else (set "READY=1")

echo.
echo ============================================================
if defined READY (
    echo   安装成功！后端已作为服务运行。
    echo.
    echo   访问地址 : http://localhost:8000     （admin / admin123）
    echo   服务管理 : services.msc  →  %SVC%，或执行 sc query %SVC%
    echo   日志文件 : %APP_DIR%\logs\service.log
    echo.
    echo   说明：服务已设为"自动"启动，电脑开机即自动运行，无需登录。
) else (
    echo   服务已注册，但自检未通过（health 未返回 ok）。诊断如下：
    "%PY%" "%APP_DIR%\tools\win_service.py" diag
)
echo ============================================================
echo.
echo 常用命令：
echo   net stop  %SVC%                          停服务
echo   net start %SVC%                          起服务
echo   deploy\uninstall_service.bat             卸载服务
echo.
pause
