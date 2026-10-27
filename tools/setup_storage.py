# -*- coding: utf-8 -*-
"""存储布局检查 / 初始化（换设备只改 .env，不改代码）

用法::

    # 只检查：打印每个目录 → 真实挂载点、设备，以及是否有问题
    python tools/setup_storage.py

    # 建目录 + 写测试文件验证可写（幂等，重复跑没事）
    python tools/setup_storage.py --apply

    # 生成 systemd drop-in（RequiresMountsFor 跟着配置走，不写死盘符）
    python tools/setup_storage.py --write-unit deploy/systemd/storage.conf
    sudo install -D -m 644 deploy/systemd/storage.conf \\
         /etc/systemd/system/labelyou-backend.service.d/storage.conf
    sudo systemctl daemon-reload && sudo systemctl restart labelyou-backend

    # 严格模式：目录落在根分区（数据盘没挂上）就算失败，退出码 1
    python tools/setup_storage.py --strict

换设备流程：改 `.env` 里的 STORAGE_ROOT / RUNS_DIR / ARCHIVE_DIR / DATABASE_URL
→ `python tools/setup_storage.py --apply --write-unit ...` → 装 drop-in → 重启服务。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.config import settings  # noqa: E402
from server.services import storage_layout  # noqa: E402


def _db_file() -> Path | None:
    from server.database import _resolve_database_url

    url = _resolve_database_url()
    if not url.startswith("sqlite:///"):
        return None
    return Path(url[len("sqlite:///"):])


def cmd_apply() -> int:
    """按配置建目录，并写一个测试文件确认真的可写（盘没挂上时能立刻发现）。"""
    targets = [
        ("uploads", settings.upload_path),
        ("datasets", settings.dataset_path),
        ("exports", settings.export_path),
        ("runs", settings.runs_path),
        ("archive", settings.archive_path),
        ("db_backups", settings.archive_path / "db_backups"),
    ]
    db_file = _db_file()
    if db_file is not None:
        targets.append(("database", db_file.parent))

    failed = 0
    for name, path in targets:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            print(f"  [OK] {name:<11} {path}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [!!] {name:<11} {path}  ->  {exc}")
    return 1 if failed else 0


def cmd_report(strict: bool) -> int:
    print("存储布局（全部来自 .env / 环境变量）")
    print(f"  STORAGE_ROOT = {settings.STORAGE_ROOT}")
    print(f"  RUNS_DIR     = {settings.RUNS_DIR}")
    print(f"  ARCHIVE_DIR  = {settings.ARCHIVE_DIR}")
    print(f"  DATABASE_URL = {settings.DATABASE_URL or '(默认 <STORAGE_ROOT>/app.db)'}")
    print()
    print("目录 → 挂载点")
    for line in storage_layout.summary_lines():
        print(f"  {line}")
    print()
    issues = storage_layout.problems(strict=strict)
    if issues:
        print("问题：")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("检查通过：目录存在、可写" + ("、都在独立挂载点上" if strict else ""))
    print()
    print(storage_layout.systemd_dropin(), end="")
    print()
    for hint in storage_layout.fstab_hints():
        print(f"  {hint}")
    return 1 if issues else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="存储布局检查 / 初始化")
    ap.add_argument("--apply", action="store_true", help="按配置创建目录并验证可写")
    ap.add_argument("--write-unit", metavar="PATH", help="把 systemd drop-in 写到指定文件")
    ap.add_argument("--strict", action="store_true", help="目录落在根分区也算失败")
    args = ap.parse_args()

    code = 0
    if args.apply:
        print("初始化目录：")
        code |= cmd_apply()
        print()

    code |= cmd_report(strict=args.strict)

    if args.write_unit:
        out = Path(args.write_unit)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(storage_layout.systemd_dropin(), encoding="utf-8")
        print()
        print(f"已写出 drop-in: {out}")
        print("安装：")
        print(
            "  sudo install -D -m 644 "
            f"{out} /etc/systemd/system/labelyou-backend.service.d/storage.conf"
        )
        print("  sudo systemctl daemon-reload && sudo systemctl restart labelyou-backend")

    return code


if __name__ == "__main__":
    raise SystemExit(main())
