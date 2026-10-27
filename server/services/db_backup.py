# -*- coding: utf-8 -*-
"""
SQLite 主库自动备份
===================
原图放机械盘、数据库跑在 SSD 上时，标注数据只有一份。本模块把 app.db
定期用 sqlite3 在线备份 API 复制到 `<ARCHIVE_DIR>/db_backups/`：

- 备份文件：`app-YYYYmmdd-HHMMSS.db`（可用任意 SQLite 工具直接打开）
- 启动先备份一次，之后每 `DB_BACKUP_INTERVAL_H` 小时一次
- 只保留最近 `DB_BACKUP_KEEP` 份（旧文件自动删除）
- 只在 DATABASE_URL 指向本地 sqlite 文件时生效；外接数据库跳过
- 用 backup API 而不是 copy 文件：WAL 模式下直接 cp 会得到不一致的快照

关闭：`.env` 里 `DB_BACKUP_ENABLED=False`。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import settings

_BACKUP_PREFIX = "app-"


def _db_file() -> Optional[Path]:
    """当前引擎对应的 SQLite 文件；非 sqlite / 内存库返回 None。"""
    from ..database import engine

    url = engine.url
    try:
        if url.get_backend_name() != "sqlite":
            return None
    except Exception:  # noqa: BLE001
        return None
    database = url.database
    if not database or database == ":memory:":
        return None
    return Path(database)


def backup_dir() -> Path:
    p = settings.archive_path / "db_backups"
    p.mkdir(parents=True, exist_ok=True)
    return p


def backup_once(keep: Optional[int] = None) -> Optional[Path]:
    """立即备份一次，返回备份文件路径；跳过/失败返回 None。"""
    src_path = _db_file()
    if src_path is None or not src_path.exists():
        return None
    target_dir = backup_dir()
    dst = target_dir / f"{_BACKUP_PREFIX}{datetime.now():%Y%m%d-%H%M%S}.db"
    if dst.exists():                      # 同一秒重复触发
        dst = target_dir / f"{_BACKUP_PREFIX}{datetime.now():%Y%m%d-%H%M%S}-{int(time.time() % 1000)}.db"

    src = sqlite3.connect(str(src_path), timeout=30)
    try:
        dst_con = sqlite3.connect(str(dst))
        try:
            src.backup(dst_con)           # 在线一致性备份
        finally:
            dst_con.close()
    finally:
        src.close()

    prune(target_dir, settings.DB_BACKUP_KEEP if keep is None else keep)
    return dst


def prune(target_dir: Path, keep: int) -> list[str]:
    """只保留最近 keep 份备份，返回删掉的文件名。"""
    removed: list[str] = []
    if keep <= 0:
        return removed
    try:
        files = sorted(
            (p for p in target_dir.glob(f"{_BACKUP_PREFIX}*.db") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:  # noqa: BLE001
        return removed
    for old in files[keep:]:
        try:
            old.unlink(missing_ok=True)
            removed.append(old.name)
        except Exception:  # noqa: BLE001
            pass
    return removed


def start_background_backup() -> Optional[threading.Thread]:
    """启动后台备份线程（先立刻备一份）。未启用/非 sqlite 时返回 None。"""
    if not getattr(settings, "DB_BACKUP_ENABLED", True):
        print("  数据库备份: 已关闭（DB_BACKUP_ENABLED=False）")
        return None
    if _db_file() is None:
        print("  数据库备份: 跳过（当前不是本地 SQLite）")
        return None

    interval_s = max(0.25, float(settings.DB_BACKUP_INTERVAL_H)) * 3600

    def _loop() -> None:
        while True:
            try:
                out = backup_once()
                if out is not None:
                    print(f"[数据库备份] {out}")
            except Exception as exc:      # noqa: BLE001 - 备份失败不能影响服务
                print(f"[数据库备份] 失败: {exc}")
            time.sleep(interval_s)

    t = threading.Thread(target=_loop, name="db-backup", daemon=True)
    t.start()
    print(
        f"  数据库备份: {backup_dir()}（每 {settings.DB_BACKUP_INTERVAL_H}h，"
        f"保留 {settings.DB_BACKUP_KEEP} 份）"
    )
    return t
