# -*- coding: utf-8 -*-
"""存储布局自检 / 挂载点推导
============================
本平台的所有磁盘位置都来自配置（`.env` / 环境变量），代码和 systemd 里不写死盘符。
换设备时只需要改配置：

    STORAGE_ROOT=/data/hdd/youlobe/storage
    RUNS_DIR=/data/ssd/youlobe/runs
    ARCHIVE_DIR=/data/hdd/youlobe/archive
    DATABASE_URL=sqlite:////data/ssd/youlobe/db/app.db

然后运行 `python tools/setup_storage.py --apply --print-unit`：
- 按配置建目录
- 推导每个目录实际落在哪个挂载点，生成对应的 systemd `RequiresMountsFor`
- 提示 fstab / 目录权限该怎么配

本模块同时给 `server/main.py` 启动时做自检用：
- 默认只打印现状与警告
- `STORAGE_REQUIRE_MOUNT=True` 时，配置目录落在根分区（盘没挂上）会拒绝启动
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

# 参与自检的目录（名字, 配置键, 路径）
def target_dirs() -> list[tuple[str, str, Path]]:
    roots = [
        ("原图/数据集/导出", "STORAGE_ROOT", Path(settings.STORAGE_ROOT).expanduser().resolve()),
        ("训练工作区", "RUNS_DIR", settings.runs_path),
        ("训练归档", "ARCHIVE_DIR", settings.archive_path),
    ]
    return roots


@dataclass
class DirInfo:
    name: str
    key: str
    path: Path
    exists: bool
    writable: bool
    mountpoint: str
    device: str

    @property
    def dedicated(self) -> bool:
        """是否落在独立挂载点（不是根分区）。"""
        return self.mountpoint not in ("/", "")


def _mount_table() -> list[tuple[str, str]]:
    """[(挂载点, 设备)]；优先 /proc/mounts，取不到时退化为只有根分区。"""
    entries: list[tuple[str, str]] = []
    try:
        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    entries.append((parts[1].replace("\\040", " "), parts[0]))
    except OSError:
        pass
    if not entries:
        entries = [("/", "")]
    return entries


def mountpoint_of(path: Path) -> tuple[str, str]:
    """返回 (最深的挂载点, 设备)。"""
    try:
        p = path.resolve()
    except OSError:
        p = path
    best, best_device, best_len = "/", "", 0
    for mp, dev in _mount_table():
        try:
            mp_path = Path(mp)
            if p == mp_path or mp_path in p.parents:
                if len(mp) > best_len:
                    best, best_device, best_len = mp, dev, len(mp)
        except OSError:
            continue
    return best, best_device


def _writable(path: Path) -> bool:
    try:
        return path.is_dir() and os.access(path, os.W_OK)
    except OSError:
        return False


def describe() -> list[DirInfo]:
    out: list[DirInfo] = []
    for name, key, path in target_dirs():
        mp, dev = mountpoint_of(path)
        out.append(
            DirInfo(
                name=name, key=key, path=path,
                exists=path.is_dir(), writable=_writable(path),
                mountpoint=mp, device=dev,
            )
        )
    return out


def required_mountpoints() -> list[str]:
    seen: list[str] = []
    for info in describe():
        if info.mountpoint not in seen:
            seen.append(info.mountpoint)
    return seen


def problems(strict: bool = False) -> list[str]:
    """目录不存在 / 不可写，strict 时再加上「落在根分区」。"""
    issues: list[str] = []
    for info in describe():
        if not info.exists:
            issues.append(f"{info.key}={info.path} 不存在")
        elif not info.writable:
            issues.append(f"{info.key}={info.path} 不可写")
        elif strict and not info.dedicated:
            issues.append(
                f"{info.key}={info.path} 落在根分区 {info.mountpoint}（数据盘可能没挂载）"
            )
    return issues


def summary_lines() -> list[str]:
    lines: list[str] = []
    for info in describe():
        flag = "OK " if info.exists and info.writable else "!! "
        where = info.mountpoint if info.dedicated else "/（根分区）"
        lines.append(f"{flag}{info.key:<13} {info.path}  ← {where}")
    return lines


def systemd_dropin(unit: str = "labelyou-backend.service") -> str:
    """按当前配置生成 systemd drop-in（RequiresMountsFor 跟随配置，不写死盘符）。"""
    all_mps = required_mountpoints()
    # 根分区永远就绪，不写进 RequiresMountsFor（但上面的自检仍会提示它）
    mps = [m for m in all_mps if m not in ("/", "")] or ["/"]
    return (
        "# 由 tools/setup_storage.py 依据 .env 自动生成 —— 换设备后重新生成，不要手改\n"
        f"# unit: {unit}\n"
        "[Unit]\n"
        f"# 这些挂载点必须就绪后才启动服务：{', '.join(all_mps)}\n"
        f"RequiresMountsFor={' '.join(mps)}\n"
        "After=local-fs.target\n"
    )


def fstab_hints() -> list[str]:
    """给「换设备」时写 fstab 用的提示（不自动改 /etc/fstab）。"""
    hints = ["按上面的挂载点配置 /etc/fstab（用 UUID，nofail 避免开机卡住）："]
    for mp in required_mountpoints():
        if mp == "/":
            continue
        hints.append(f"  UUID=<blkid 查到的该分区 UUID>  {mp}  ext4  defaults,nofail,x-systemd.device-timeout=10s  0 2")
    hints.append("挂载后确认属主：sudo chown -R <运行用户>:<运行用户> <各目录>")
    return hints
