# -*- coding: utf-8 -*-
"""
日志轮转（copytruncate 风格）
=============================
平台有两类日志落盘方式，必须区别对待：

1. **Python logging 的 RotatingFileHandler**（Windows 服务 `logs/service.log`）
   —— 由 handler 自己按大小轮转，本模块 **跳过** 这些文件，避免双重轮转打架。

2. **shell 重定向**（`nohup ... >> logs/backend.log`、`start_backend.bat` 的
   `> logs\\backend.log 2>&1`）—— 进程拿着一只 append fd，Python 侧改不了它的
   轮转，所以采用 logrotate 的 copytruncate 策略：

       copy backend.log -> backend.log.1（旧 .1 顺移成 .2 …，超出份数丢弃）
       truncate backend.log 到 0

   fd 是 O_APPEND，截断后继续写会落在文件末尾（即新文件的 0 位置），
   不需要重启进程、也不会丢 fd。

后台线程每 `LOG_ROTATE_INTERVAL_S` 秒扫描一次 `logs/*.log`，超限就转。
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Iterable

log = logging.getLogger("log_rotation")

# 进程内所有轮转操作串行化（多线程同时进入会互相搬文件）
_LOCK = threading.Lock()


def _handler_managed_paths() -> set[str]:
    """找出已被 logging handler 接管的日志文件（它们的轮转交给 handler）。"""
    managed: set[str] = set()
    for h in logging.getLogger().handlers:
        base = getattr(h, "baseFilename", None)
        if base:
            try:
                managed.add(str(Path(base).resolve()))
            except Exception:  # noqa: BLE001
                continue
    return managed


def rotate_file(path: Path, max_bytes: int, backups: int = 5) -> bool:
    """把超过 max_bytes 的日志 copytruncate 轮转一次；返回是否真的转了。"""
    if backups < 1:
        backups = 1
    with _LOCK:
        try:
            if not path.is_file():
                return False
            size = path.stat().st_size
            if max_bytes > 0 and size < max_bytes:
                return False

            # 旧的备份顺移：.4 -> .5（最老的丢弃），.1 -> .2 ...
            for idx in range(backups, 0, -1):
                src = path.with_suffix(path.suffix + f".{idx}")
                if idx == backups:
                    src.unlink(missing_ok=True)
                else:
                    dst = path.with_suffix(path.suffix + f".{idx + 1}")
                    if src.exists():
                        src.replace(dst)

            dst = path.with_suffix(path.suffix + ".1")
            shutil.copyfile(path, dst)

            # 截断到 0：append fd 之后会从新文件末尾继续写
            with path.open("wb"):
                pass
            log.info("日志轮转：%s（%.1f MB -> %s）", path.name, size / 1024 / 1024, dst.name)
            return True
        except Exception as exc:  # noqa: BLE001 - 轮转失败不能影响服务
            log.warning("日志轮转失败 %s：%s", path, exc)
            return False


def rotate_dir(log_dir: Path, max_bytes: int, backups: int) -> list[str]:
    """扫描目录下所有 *.log（跳过 handler 接管的与已轮转的），超限就转。"""
    rotated: list[str] = []
    if not log_dir.is_dir():
        return rotated
    managed = _handler_managed_paths()
    for path in sorted(log_dir.glob("*.log")):
        try:
            if str(path.resolve()) in managed:
                continue                    # 交给 RotatingFileHandler
        except Exception:  # noqa: BLE001
            pass
        if rotate_file(path, max_bytes, backups):
            rotated.append(path.name)
    return rotated


class LogRotationWorker:
    """后台守护线程：定期检查并轮转日志。"""

    def __init__(self, log_dirs: Iterable[Path], max_bytes: int, backups: int, interval_s: int = 60):
        self.log_dirs = [Path(p) for p in log_dirs]
        self.max_bytes = max_bytes
        self.backups = backups
        self.interval_s = max(10, int(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self) -> list[str]:
        rotated: list[str] = []
        for d in self.log_dirs:
            rotated += rotate_dir(d, self.max_bytes, self.backups)
        return rotated

    def _run(self) -> None:
        # 启动先查一次：上次跑挂时可能已经超限很久了
        try:
            self.tick()
        except Exception:  # noqa: BLE001
            pass
        while not self._stop.wait(self.interval_s):
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                pass

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="log-rotation")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread = None


def log_dir_candidates(logs_dir: Path) -> list[Path]:
    """只需要一个目录：<repo>/logs（service.log / backend.log 都在这里）。"""
    return [Path(logs_dir)]


def backups_info(log_dir: Path) -> list[dict]:
    """列出已轮转的备份文件（给运行监控页展示）。"""
    out: list[dict] = []
    if not log_dir.is_dir():
        return out
    for path in sorted(log_dir.glob("*.log.*")):
        try:
            st = path.stat()
        except OSError:
            continue
        out.append({
            "file": path.name,
            "size": st.st_size,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
        })
    return out
