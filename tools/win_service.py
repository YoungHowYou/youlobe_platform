# -*- coding: utf-8 -*-
"""
Windows 服务封装（pywin32）
==========================
把平台后端注册成真正的 Windows 服务，由 SCM 托管：

    开机自启（无需登录）· 崩溃自动重启 · services.msc 可管理 · 无控制台窗口

服务标识：
    服务名     LabelyouBackend
    显示名     Labelyou 缺陷标注平台后端
    服务镜像   "<项目>\\venv\\Scripts\\python.exe" "<项目>\\tools\\win_service.py" service
    监听       0.0.0.0:8000（可用环境变量 LABELYOU_HOST / LABELYOU_PORT 覆盖）
    日志       logs\\service.log（RotatingFileHandler，默认 20MB × 5，可用 LOG_MAX_MB / LOG_BACKUP_COUNT 覆盖）+ logs\\service.stdout.log
               其它 logs/*.log 由 app 内 copytruncate 轮转（server/services/log_rotation.py，60s 检查）

    [!] 服务镜像刻意用 **venv 的 python.exe**，而不是 pywin32 自带的 pythonservice.exe。
      后者按注册表定位基础解释器且不处理 .pth，连它引导时要用的 servicemanager 都导入不了
      （现象：SCM 7009 连接超时 + 日志空白）。详见 install_service() 的注释与 SKILL.md。

命令（install / remove / start / stop **必须管理员权限的终端**）：
    venv\\Scripts\\python.exe tools\\win_service.py install
    venv\\Scripts\\python.exe tools\\win_service.py start
    venv\\Scripts\\python.exe tools\\win_service.py stop
    venv\\Scripts\\python.exe tools\\win_service.py restart
    venv\\Scripts\\python.exe tools\\win_service.py remove

前台调试 / 排查：
    venv\\Scripts\\python.exe tools\\win_service.py console       # 前台跑，Ctrl+C 退出
    venv\\Scripts\\python.exe tools\\win_service.py diag          # 服务起不来时一键诊断
    venv\\Scripts\\python.exe tools\\win_service.py wait-health   # 轮询 /api/health
    venv\\Scripts\\python.exe tools\\win_service.py free-port     # 腾出 8000 端口

    `service` 是给 SCM 用的入口，手动运行没有意义。

一键脚本：deploy\\install_service.bat（装）/ deploy\\uninstall_service.bat（卸）
"""

import os
import sys
import threading
import time
import traceback
from pathlib import Path

# ----------------------------------------------------------------------------
# 启动引导：服务的工作目录默认是 C:\Windows\System32，
# 而 .env 里的 STORAGE_ROOT=./storage 是相对路径，必须先切到项目根目录。
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ----------------------------------------------------------------------------
# 关键：让「服务宿主进程」能找到这个 venv 里的包
# ----------------------------------------------------------------------------
# pywin32 的 pythonservice.exe 是按注册表 PythonCore 定位解释器的，它**不认识 venv**：
# 用系统解释器启动时 venv 的 site-packages 不在 sys.path 上，于是连
# `import servicemanager` 都会 ModuleNotFoundError —— 服务表现为事件日志 7009
# （等待连接超时 = 1053），且 logs\service.log 一行都不写。
#
# 基础解释器与 venv 是同一个 3.10（cp310 扩展可直接加载），所以这里把 venv 的
# site-packages 布局手工补进 sys.path 与 DLL 搜索路径，宿主就能正常导入
# pywin32（servicemanager / win32serviceutil）以及 fastapi / torch。
_VENV_SITE = ROOT / "venv" / "Lib" / "site-packages"
if _VENV_SITE.is_dir():
    for _p in (_VENV_SITE / "win32", _VENV_SITE / "win32" / "lib", _VENV_SITE,
               _VENV_SITE / "pywin32_system32"):
        if _p.is_dir():
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
            try:
                os.add_dll_directory(str(_p))       # 让 pywintypes3XX.dll 等能被找到
            except (AttributeError, OSError):
                pass

# 防止 ultralytics 在服务环境里尝试联网安装依赖（会卡住启动）
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 控制台可能是 GBK 代码页，遇到无法编码的字符（如 emoji）会让 print 直接抛
# UnicodeEncodeError 把命令打断。这里把无法编码的字符降级成 "?"，保证输出不中断。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:                                    # noqa: BLE001
        pass

SERVICE_NAME = "LabelyouBackend"
SERVICE_DISPLAY_NAME = "Labelyou 缺陷标注平台后端"
SERVICE_DESCRIPTION = (
    "Labelyou 工业缺陷检测平台后端：FastAPI API + 前端静态托管 + 进程内训练线程。"
    "使用本地 SQLite 存储（storage/app.db），默认监听 8000 端口，无需 MySQL/Redis/Celery。"
)

HOST = os.environ.get("LABELYOU_HOST", "0.0.0.0")
PORT = int(os.environ.get("LABELYOU_PORT", "8000"))
LOG_DIR = ROOT / "logs"


def setup_file_logging(stream: bool = False) -> None:
    """服务没有控制台，日志必须落到文件；控制台模式额外打屏。

    轮转参数统一读 settings.LOG_MAX_MB / LOG_BACKUP_COUNT（默认 20MB × 5），
    和 app 内部的 copytruncate 轮转（server/services/log_rotation.py）保持同一口径；
    service.log 由本 handler 自己轮转，app 的 worker 会识别并跳过它。
    """
    import logging
    from logging.handlers import RotatingFileHandler

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    try:
        from server.config import settings as _settings
        max_bytes = max(1, int(_settings.LOG_MAX_MB)) * 1024 * 1024
        backup_count = max(1, int(_settings.LOG_BACKUP_COUNT))
    except Exception:                      # 配置读不到时回到历史默认值
        max_bytes, backup_count = 8 * 1024 * 1024, 5

    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        fh = RotatingFileHandler(LOG_DIR / "service.log", maxBytes=max_bytes,
                                 backupCount=backup_count, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s"))
        root.addHandler(fh)

    if stream and not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        root.addHandler(sh)


class BackendRunner:
    """把 uvicorn 跑在子线程里，便于服务主线程等待停止信号。

    放在子线程还有一个好处：uvicorn 的 `capture_signals()` 检测到非主线程会直接跳过
    安装信号处理器，服务进程里没有控制台信号，正好不会踩坑。
    """

    def __init__(self, host: str = HOST, port: int = PORT):
        self.host = host
        self.port = port
        self.server = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        import uvicorn
        from server.main import app

        # log_config=None：不覆盖 logging 配置，让 uvicorn 的日志走我们挂的 root handler
        config = uvicorn.Config(app, host=self.host, port=self.port,
                                log_level="info", log_config=None)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="uvicorn", daemon=True)
        self.thread.start()

    def wait_ready(self, timeout: float = 120.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.server is not None and self.server.started:
                return True
            if self.thread is not None and not self.thread.is_alive():
                return False          # uvicorn 提前退出（端口占用等）
            time.sleep(0.2)
        return False

    def stop(self, timeout: float = 30.0) -> None:
        if self.server is not None:
            self.server.should_exit = True          # 触发 uvicorn 优雅退出
        if self.thread is not None:
            self.thread.join(timeout=timeout)


# ----------------------------------------------------------------------------
# SCM 服务实现
# ----------------------------------------------------------------------------
try:
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil
    HAS_PYWIN32 = True
except ImportError:                                 # console 模式不需要 SCM 相关模块
    HAS_PYWIN32 = False


if HAS_PYWIN32:

    class BackendService(win32serviceutil.ServiceFramework):
        _svc_name_ = SERVICE_NAME
        _svc_display_name_ = SERVICE_DISPLAY_NAME
        _svc_description_ = SERVICE_DESCRIPTION

        def __init__(self, args):
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.runner: BackendRunner | None = None

        def SvcStop(self):
            """SCM 要求停止：先上报 STOP_PENDING（避免 1053），再让 uvicorn 优雅退出。"""
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            if self.runner is not None:
                self.runner.server.should_exit = True
            win32event.SetEvent(self.stop_event)

        def SvcDoRun(self):
            import logging

            setup_file_logging()
            log = logging.getLogger("service")
            log.info("=" * 70)
            log.info("%s 启动中  cwd=%s", SERVICE_NAME, os.getcwd())
            # 宿主进程（pythonservice.exe）用的解释器与 sys.path 不一定是我们期望的，
            # 服务起不来时这几行是唯一线索，务必保留。
            log.info("executable=%s", sys.executable)
            log.info("prefix=%s  base_prefix=%s  exec_prefix=%s",
                     sys.prefix, sys.base_prefix, sys.exec_prefix)
            log.info("sys.path[0:%d]=%s", min(8, len(sys.path)), sys.path[:8])
            try:
                self.runner = BackendRunner()
                self.runner.start()
                if not self.runner.wait_ready():
                    raise RuntimeError(
                        f"后端未能在 120 秒内监听 {HOST}:{PORT}（端口被占用？依赖没装全？）"
                    )
                log.info("已就绪： http://%s:%s   （检测到 GPU 时训练自动走显卡）",
                         "localhost" if HOST == "0.0.0.0" else HOST, PORT)
                servicemanager.LogInfoMsg(f"{SERVICE_NAME} listening on {HOST}:{PORT}")

                # 等 SCM 的停止信号；训练线程等资源由任务的 finally 负责回收
                win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)
                log.info("收到停止信号，正在优雅关闭（未跑完的训练任务下次启动会标记为 failed）...")
            except Exception:
                log.error("服务异常：\n%s", traceback.format_exc())
                servicemanager.LogErrorMsg(traceback.format_exc())
            finally:
                if self.runner is not None:
                    self.runner.stop()
                log.info("%s 已停止", SERVICE_NAME)

    def _announce_install() -> None:
        """pywin32 标准 install 路径的提示（本项目不用，保留给 --help 场景）。"""
        from win32serviceutil import LocatePythonServiceExe
        print(f"  项目目录 : {ROOT}")
        print(f"  服务宿主 : {LocatePythonServiceExe()}")
        print(f"  日志文件 : {LOG_DIR / 'service.log'}")
        print(f"  访问地址 : http://localhost:{PORT}")


def run_console() -> int:
    """前台运行（调试用）：Ctrl+C 退出。"""
    import logging

    setup_file_logging(stream=True)
    log = logging.getLogger("console")
    log.info("控制台模式启动： http://%s:%s", "localhost" if HOST == "0.0.0.0" else HOST, PORT)

    runner = BackendRunner()
    runner.start()
    if not runner.wait_ready():
        log.error("启动失败：%s:%s 未就绪，请检查端口占用与依赖", HOST, PORT)
        runner.stop()
        return 1

    log.info("已就绪，按 Ctrl+C 退出")
    try:
        while runner.thread is not None and runner.thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("收到 Ctrl+C，正在关闭 ...")
    finally:
        runner.stop()
    return 0


def free_port(port: int = PORT) -> int:
    """停掉占用指定端口的进程——安装服务前腾地方（手动启动的实例会一直占着）。"""
    import subprocess

    try:
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True,
                             text=True, encoding="utf-8", errors="ignore").stdout
    except OSError as exc:
        print(f"  无法执行 netstat：{exc}")
        return 0

    pids: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        # 形如： TCP    0.0.0.0:8000    0.0.0.0:0    LISTENING    12345
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING":
            if parts[1].rsplit(":", 1)[-1] == str(port):
                pids.add(parts[4])

    mine = str(os.getpid())
    for pid in sorted(pids - {mine}):
        print(f"  端口 {port} 被 PID={pid} 占用，正在停止 ...")
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    if not pids - {mine}:
        print(f"  端口 {port} 空闲")
    return 0


def install_service() -> int:
    """注册服务 —— **直接用 venv 的 python.exe 当服务镜像**。

    为什么不用 pywin32 自带的 pythonservice.exe（踩过的坑，详见 SKILL.md）：
    那个宿主按注册表 PythonCore 定位解释器，用的是**基础解释器**、而且**不处理 .pth**，
    结果连它自己引导时要 import 的 `servicemanager` 都找不到（该模块在
    `venv\\Lib\\site-packages\\win32\\`），服务表现为 SCM 7009「连接超时」+ 日志空白。
    而我们无法在它 import 之前干预 sys.path（PYTHONPATH / 注册表 PythonPath 都试过，
    前者有效但服务环境拿不到，后者 CPython 已不再读）。

    直接把服务镜像设成 venv 的 python.exe 就绕开了整件事：解释器就是平时验证过能跑的那个，
    pywin32 与 fastapi/torch 都在它的 site-packages 里，什么都不用补。
    """
    if not HAS_PYWIN32:
        print("缺少 pywin32，请先执行： venv\\Scripts\\python.exe -m pip install pywin32")
        return 1

    import win32service

    python_exe = ROOT / "venv" / "Scripts" / "python.exe"
    if not python_exe.is_file():
        print(f"[错误] 找不到 venv 解释器：{python_exe}")
        print("       请先在项目根目录执行 python tools\\setup_env.py")
        return 1

    script = Path(__file__).resolve()
    exe_args = f'"{script}" service'

    # 旧的注册可能是上一个版本用 pythonservice.exe 建的，先删掉确保 binPath 被更新
    try:
        win32serviceutil.RemoveService(SERVICE_NAME)
        print(f"  已移除旧的 {SERVICE_NAME} 注册")
    except Exception:                                    # noqa: BLE001
        pass

    try:
        win32serviceutil.InstallService(
            "win_service.BackendService",
            SERVICE_NAME,
            SERVICE_DISPLAY_NAME,
            startType=win32service.SERVICE_AUTO_START,
            exeName=str(python_exe),
            exeArgs=exe_args,
            description=SERVICE_DESCRIPTION,
        )
    except Exception as exc:                             # noqa: BLE001
        print(f"[错误] 注册服务失败：{exc}")
        print("       最常见原因：没有以管理员身份运行（注册服务需要管理员权限）。")
        print("       请右键 deploy\\install_service.bat → 以管理员身份运行。")
        return 1

    print(f"  服务镜像 : \"{python_exe}\" {exe_args}")
    print("  启动方式 : 自动（SYSTEM 身份，开机即启，无需登录）")
    print(f"  日志文件 : {LOG_DIR / 'service.log'}")
    print(f"  访问地址 : http://localhost:{PORT}")
    return 0


def run_as_service() -> int:
    """由 SCM 启动时的入口（服务镜像 = venv 的 python.exe + 本脚本 + service 参数）。"""
    if not HAS_PYWIN32:
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # 服务进程没有控制台：stdout/stderr 是无效句柄（sys.stdout 会是 None），
    # 任何库的一次 print 都会抛异常。统一兜到文件里。
    try:
        stream = open(LOG_DIR / "service.stdout.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = stream
        sys.stderr = stream
    except OSError:
        pass

    setup_file_logging()
    import logging
    logging.getLogger("service").info(
        "交给 SCM 调度： %s service   python=%s", Path(__file__).resolve(), sys.executable)

    servicemanager.Initialize()
    servicemanager.PrepareToHostSingle(BackendService)
    try:
        # 阻塞在这里，直到服务停止。SCM 调用 ServiceMain → ServiceFramework.SvcRun()
        # → 先上报 SERVICE_RUNNING，再执行 SvcDoRun()（所以启动慢也不会 1053）。
        servicemanager.StartServiceCtrlDispatcher()
    except Exception as exc:                             # noqa: BLE001
        logging.getLogger("service").error(
            "无法连接服务控制器（正常现象：本命令只有由 SCM 启动时才有效）：%s", exc)
        print(f"[提示] 本命令需由 Windows 服务控制器启动。手动前台运行请用 `console` 子命令。\n       {exc}")
        return 1
    return 0


FIREWALL_RULE = "LabelyouBackend-8000"


def _is_admin() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:                                    # noqa: BLE001
        return False


def _lan_ips() -> list:
    import socket

    ips = set()
    for host in (socket.gethostname(), socket.getfqdn()):
        try:
            for info in socket.getaddrinfo(host, None, socket.AF_INET):
                ips.add(info[4][0])
        except OSError:
            pass
    return sorted(ip for ip in ips if not ip.startswith(("127.", "169.254.")))


def _run_ps(script: str):
    import subprocess

    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )


def open_firewall() -> int:
    """放行 8000 端口的入站访问（仅本地子网），让局域网内其它电脑能打开本平台。

    用 PowerShell 的 New-NetFirewallRule，**不经过 cmd**——netsh 写在 .bat 里
    会出现输出被当成命令执行的解析怪癖。
    """
    if not _is_admin():
        print("[错误] 需要管理员权限。")
        print("       请右键 deploy\\open_firewall.bat → 以管理员身份运行；")
        print("       或右键 deploy\\install_service.bat → 以管理员身份运行（它已包含这一步）。")
        return 1

    exists = _run_ps(
        f"if (Get-NetFirewallRule -DisplayName '{FIREWALL_RULE}' "
        f"-ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}"
    )
    if exists.returncode == 0:
        print(f"[防火墙] 规则 {FIREWALL_RULE} 已存在，跳过")
    else:
        print(f"[防火墙] 添加放行规则 {FIREWALL_RULE}（TCP {PORT}，仅本地子网）")
        res = _run_ps(
            f"New-NetFirewallRule -DisplayName '{FIREWALL_RULE}' -Direction Inbound "
            f"-Action Allow -Protocol TCP -LocalPort {PORT} -RemoteAddress LocalSubnet "
            f"-Profile Any -Description 'Labelyou Web port {PORT}' | Out-Null"
        )
        if res.returncode != 0:
            print(f"[错误] 添加失败：{(res.stderr or res.stdout or '').strip()}")
            return 1
        print("[防火墙] 已添加")

    print("\n局域网内其它电脑可以这样访问本平台：")
    ips = _lan_ips()
    if ips:
        for ip in ips:
            print(f"  http://{ip}:{PORT}")
    else:
        print("  （没找到局域网 IP，检查网线/WiFi 是否已连接）")
    print(f"\n本机访问： http://localhost:{PORT}")
    print("默认账号： admin / admin123")
    print("\n说明：规则只放行本地子网，公网来源仍被拦截。")
    print("      若本机 IP 由 DHCP 分配，换租约后地址可能变，建议在路由器上给这台机器保留固定 IP。")
    return 0


def close_firewall() -> int:
    """移除放行规则（卸载时用）。"""
    if not _is_admin():
        print("[错误] 需要管理员权限才能移除防火墙规则。")
        return 1
    res = _run_ps(
        f"Remove-NetFirewallRule -DisplayName '{FIREWALL_RULE}' -ErrorAction SilentlyContinue"
    )
    print(f"[防火墙] 已尝试移除规则 {FIREWALL_RULE}（本来不存在则忽略）")
    return 0 if res.returncode == 0 else 1


def diag() -> int:
    """服务起不来时的诊断：一次把该看的都打出来。"""
    import subprocess

    def run_ps(script: str) -> str:
        try:
            return subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True, text=True, encoding="utf-8", errors="ignore",
            ).stdout.strip()
        except OSError as exc:
            return f"<执行失败: {exc}>"

    print("=" * 70)
    print(f"服务名     : {SERVICE_NAME}")
    print(f"项目根目录 : {ROOT}")
    print(f"当前解释器 : {sys.executable}")
    print(f"venv 站点包: {_VENV_SITE}  {'存在' if _VENV_SITE.is_dir() else '缺失'}")

    print("\n--- 服务状态 ---")
    print(run_ps(f"sc.exe query {SERVICE_NAME}") or "<查不到该服务：尚未安装>")

    print("\n--- 服务镜像（binPath）---")
    qc = run_ps(f"sc.exe qc {SERVICE_NAME}")
    if qc:
        line = next((l for l in qc.splitlines() if "BINARY_PATH_NAME" in l), "")
        binpath = line.split(":", 1)[1].strip() if ":" in line else ""
        print(binpath or qc)
        if "pythonservice" in binpath.lower():
            print("  [!] 这是旧版注册（pywin32 宿主）。宿主按注册表用基础解释器且不处理 .pth，")
            print("    连它自己的 servicemanager 都导入不了 → 7009 超时、日志空白。")
            print("    重新以管理员身份运行 deploy\\install_service.bat 即可换成 venv 解释器。")
        else:
            python_exe = ROOT / "venv" / "Scripts" / "python.exe"
            script = Path(__file__).resolve()
            print(f"  {python_exe}  {'存在' if python_exe.is_file() else '[缺失]'}")
            print(f"  {script}  {'存在' if script.is_file() else '[缺失]'}")
    else:
        print("<查不到该服务：尚未安装，请以管理员身份运行 deploy\\install_service.bat>")

    print("\n--- 事件日志：Python Service 报错（最近 5 条）---")
    events = run_ps(
        "Get-WinEvent -LogName Application -MaxEvents 300 -ErrorAction SilentlyContinue | "
        "Where-Object { $_.ProviderName -eq 'Python Service' } | Select-Object -First 5 | "
        "ForEach-Object { $_.TimeCreated.ToString('MM-dd HH:mm:ss') + '  ' + "
        "(($_.Properties | ForEach-Object { $_.Value }) -join ' ') }"
    )
    print(events or "<没有记录（说明宿主还没跑到 Python 初始化）>")

    print("\n--- 事件日志：SCM 报错（最近 5 条）---")
    scm = run_ps(
        "Get-WinEvent -LogName System -MaxEvents 500 -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.ProviderName -eq 'Service Control Manager' -and $_.Message -match '{SERVICE_NAME}' }} | "
        "Select-Object -First 5 | ForEach-Object { $_.TimeCreated.ToString('MM-dd HH:mm:ss') + "
        "'  [' + $_.Id + '] ' + ($_.Message -replace \"`r`n\", ' ') }"
    )
    print(scm or "<无>")

    print(f"\n--- {LOG_DIR / 'service.log'} 末尾 ---")
    log = LOG_DIR / "service.log"
    if log.is_file():
        print("\n".join(log.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:]))
    else:
        print("<日志文件不存在>")

    print("\n常见结论：")
    print("  · binPath 里是 pythonservice.exe → 旧版注册，重跑 install_service.bat 换成 venv 解释器")
    print("  · 事件里有 ModuleNotFoundError → 解释器找错了（现版本不会再出现）")
    print("  · 7009 超时且无 Python 报错 → 服务进程压根没起来，先看 binPath 指的解释器是否存在")
    print("  · 日志有 '后端未能在 120 秒内监听' → 8000 端口被占用（先跑 free-port）")
    print("=" * 70)
    return 0


def wait_health(timeout: float = 60.0) -> int:
    """轮询 /api/health，就绪返回 0，超时返回 1（给安装脚本做自检用）。"""
    import json
    import urllib.error
    import urllib.request

    probe_host = "127.0.0.1" if HOST in ("0.0.0.0", "") else HOST
    url = f"http://{probe_host}:{PORT}/api/health"
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if json.load(resp).get("status") == "ok":
                    print(f"[OK] {url} 已就绪")
                    return 0
        except Exception as exc:                       # noqa: BLE001
            last = str(exc)
        time.sleep(1)
    print(f"[FAIL] {timeout:.0f} 秒内 {url} 未返回 ok（最后错误：{last}）")
    return 1


def main() -> None:
    # 命令可能带前置选项（如 --startup auto install），所以要在整个 argv 里找
    args = [a.lower() for a in sys.argv[1:]]

    # console 是我们的自定义命令，不进 SCM 那套流程
    if "console" in args:
        raise SystemExit(run_console())

    if "service" in args:
        raise SystemExit(run_as_service())

    if "wait-health" in args:
        raise SystemExit(wait_health())

    if "diag" in args:
        raise SystemExit(diag())

    if "free-port" in args:
        raise SystemExit(free_port())

    if "open-firewall" in args:
        raise SystemExit(open_firewall())

    if "close-firewall" in args:
        raise SystemExit(close_firewall())

    if not HAS_PYWIN32:
        print("缺少 pywin32。请先执行：")
        print(r"    venv\Scripts\python.exe -m pip install pywin32")
        raise SystemExit(1)

    # install/update/reinstall 走我们自己的实现（服务镜像是 venv 的 python.exe，
    # 不是 pywin32 的 pythonservice.exe），其余命令交给 pywin32 标准实现。
    if {"install", "update", "reinstall"} & set(args):
        raise SystemExit(install_service())

    win32serviceutil.HandleCommandLine(BackendService)


if __name__ == "__main__":
    main()
