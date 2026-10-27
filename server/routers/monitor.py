# -*- coding: utf-8 -*-
"""运行监控 API：实时日志 + 当前活动 / 资源状态。

给前端「运行监控」页提供两样东西：
  - /api/monitor/status：现在在干什么（活动任务、CPU/内存/磁盘、GPU、进程）
  - /api/monitor/logs  ：后端日志尾部（带「仅业务日志」过滤，去掉 HTTP 访问行）
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path

import psutil
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models.train_task import TrainTask
from ..services.log_rotation import backups_info
from ..tasks.train_task import training_dispatcher
from .train import gpu_stats

router = APIRouter(prefix="/api/monitor", tags=["运行监控"])

_ROOT = Path(__file__).resolve().parents[2]
_LOG_CANDIDATES = (_ROOT / "logs" / "backend.log", _ROOT / "logs" / "service.log")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# uvicorn 访问日志：'INFO:     192.168.1.2:5678 - "GET /api/xx HTTP/1.1" 200 OK'
_ACCESS_RE = re.compile(r'^\S+:\s+\S+ - "(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS) ')

_ACTIVE_STATUS = ("pending", "preparing", "training", "stopping", "exporting")


def _log_path() -> Path | None:
    for p in _LOG_CANDIDATES:
        if p.is_file():
            return p
    return None


def _tail(path: Path, max_lines: int, app_only: bool, max_bytes: int = 512 * 1024) -> list[str]:
    """从文件尾部读最多 max_bytes，返回最后 max_lines 行（已去 ANSI）。"""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - max_bytes)
        f.seek(start)
        raw = f.read()

    text = raw.decode("utf-8", errors="replace")
    if start > 0:
        text = text.split("\n", 1)[-1]  # 丢掉被截断的半行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [_ANSI_RE.sub("", line).rstrip() for line in text.split("\n")]
    lines = [line for line in lines if line.strip()]
    if app_only:
        lines = [line for line in lines if not _ACCESS_RE.match(line)]
    return lines[-max_lines:]


@router.get("/logs")
def monitor_logs(
    lines: int = Query(200, ge=10, le=2000),
    app_only: bool = Query(True),
):
    """后端日志尾部。默认过滤掉 HTTP 访问日志，只看业务/训练输出。"""
    path = _log_path()
    if path is None:
        return {"file": None, "size": 0, "mtime": None, "lines": [], "message": "未找到日志文件"}

    try:
        stat = path.stat()
        return {
            "file": str(path.relative_to(_ROOT)) if path.is_relative_to(_ROOT) else str(path),
            "size": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "lines": _tail(path, lines, app_only),
        }
    except Exception as exc:  # noqa: BLE001 - 日志读取失败不应影响页面
        return {"file": str(path), "size": 0, "mtime": None, "lines": [], "message": f"读取日志失败: {exc}"}


@router.get("/status")
def monitor_status(db: Session = Depends(get_db)):
    """当前活动任务 + 排队队列 + CPU/内存/磁盘/GPU/进程信息。"""
    tasks = (
        db.query(TrainTask)
        # pending 已单独归入 queue（它是“排队中”而不是“正在跑”），
        # 否则训练进度卡片会把下一个待跑的任务当成当前任务。
        .filter(TrainTask.status.in_([s for s in _ACTIVE_STATUS if s != "pending"]))
        .order_by(TrainTask.id.desc())
        .all()
    )

    # ---- 排队队列（内存 FIFO，顺序就是即将执行的顺序）----
    queue_ids = training_dispatcher.queue_snapshot()
    queue: list[dict] = []
    if queue_ids:
        rows = {
            t.id: t
            for t in db.query(TrainTask).filter(TrainTask.id.in_(queue_ids)).all()
        }
        for pos, tid in enumerate(queue_ids, start=1):
            t = rows.get(tid)
            if t is None:
                continue
            queue.append({
                "position": pos,
                "id": t.id,
                "project_id": t.project_id,
                "task_name": t.task_name,
                "epochs": t.epochs,
                "model_name": (t.config or {}).get("model_name"),
                "created_at": t.created_at.isoformat(timespec="seconds") if t.created_at else None,
            })

    mem = psutil.virtual_memory()

    def _disk_usage(path) -> dict | None:
        try:
            d = psutil.disk_usage(str(path))
            return {
                "free_gb": round(d.free / 1024 ** 3, 1),
                "total_gb": round(d.total / 1024 ** 3, 1),
                "used_gb": round(d.used / 1024 ** 3, 1),
                "path": str(path),
            }
        except Exception:  # noqa: BLE001
            return None

    # 三块盘可能各自独立：原图/导出（storage）、训练工作区（runs）、训练归档（archive）
    disk = _disk_usage(settings.STORAGE_ROOT)
    disk_runs = _disk_usage(settings.runs_path)
    disk_archive = _disk_usage(settings.archive_path)

    proc = psutil.Process(os.getpid())
    log = _log_path()
    log_stat = None
    if log is not None:
        try:
            st = log.stat()
            log_stat = {
                "file": str(log.relative_to(_ROOT)) if log.is_relative_to(_ROOT) else str(log),
                "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                # 轮转信息：超过 max_bytes 会自动切成 <name>.1/.2…，保留 backup_count 份
                "max_bytes": max(1, settings.LOG_MAX_MB) * 1024 * 1024,
                "backup_count": max(1, settings.LOG_BACKUP_COUNT),
                "backups": [
                    {**b, "file": f"logs/{b['file']}"} for b in backups_info(_ROOT / "logs")
                ],
            }
        except Exception:  # noqa: BLE001
            log_stat = None

    # ---- 训练进度：已用时间 / 平均每轮 / 预计剩余 ----
    now = datetime.now()
    active_tasks = []
    for t in tasks:
        elapsed = None
        if t.started_at:
            elapsed = max(0, int((now - t.started_at).total_seconds()))
        avg_epoch = None
        if elapsed is not None and (t.current_epoch or 0) > 0:
            avg_epoch = elapsed / t.current_epoch
        eta = None
        if avg_epoch is not None and t.epochs:
            eta = max(0, int(avg_epoch * max(0, t.epochs - t.current_epoch)))
        active_tasks.append({
            "id": t.id,
            "project_id": t.project_id,
            "task_name": t.task_name,
            "status": t.status,
            "current_epoch": t.current_epoch,
            "epochs": t.epochs,
            "model_name": (t.config or {}).get("model_name"),
            "started_at": t.started_at.isoformat() if t.started_at else None,
            "elapsed_s": elapsed,
            "avg_epoch_s": round(avg_epoch, 1) if avg_epoch else None,
            "eta_s": eta,
            "pct": round((t.current_epoch or 0) / t.epochs * 100, 1) if t.epochs else 0,
        })

    # 最近输出：直接告诉前端“现在在干嘛”（已过滤 HTTP 访问日志）
    recent_lines = _tail(log, 8, True) if log is not None else []

    return {
        "now": now.isoformat(timespec="seconds"),
        "active_tasks": active_tasks,
        "queue": queue,
        "dispatcher_task_id": training_dispatcher.current_task_id,
        "cpu_pct": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(),
        "mem": {
            "used_mb": round(mem.used / 1024 / 1024),
            "total_mb": round(mem.total / 1024 / 1024),
            "pct": mem.percent,
        },
        "disk": disk,
        "disk_runs": disk_runs,
        "disk_archive": disk_archive,
        "process": {
            "pid": proc.pid,
            "uptime_s": int(time.time() - proc.create_time()),
            "rss_mb": round(proc.memory_info().rss / 1024 / 1024),
        },
        "gpu": gpu_stats(),
        "log": log_stat,
        "recent_lines": recent_lines,
    }
