# -*- coding: utf-8 -*-
"""本地预训练模型（.pt）来源管理：扫描 / 路径校验 / 元信息探查。

只允许使用服务器上约定目录里的模型，避免通过 API 读取任意路径：
  - <repo>/weights/           （ultralytics 的 weights_dir）
  - <repo>/                   （直接下载常落在仓库根）
  - <STORAGE_ROOT>/models/    （大盘上的集中存放目录）
  - <repo>/storage/models/    （老布局兼容，不存在则忽略）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..config import settings

_ROOT = Path(__file__).resolve().parents[2]


def _model_dirs() -> list[Path]:
    """每次现算：STORAGE_ROOT 可能指向另一块盘，且模型目录可能后建。"""
    dirs = [_ROOT / "weights", _ROOT]
    try:
        dirs.append(Path(settings.STORAGE_ROOT).expanduser().resolve() / "models")
    except Exception:  # noqa: BLE001
        pass
    dirs.append(_ROOT / "storage" / "models")
    return dirs


# ultralytics 的 task 名称 → 平台 task_type
TASK_ALIASES = {
    "segment": "seg",
    "detect": "det",
    "classify": "cls",
    "obb": "obb",
}


def allowed_dirs() -> list[Path]:
    return [d for d in _model_dirs() if d.is_dir()]


def resolve_local_model(path: str | Path) -> Path:
    """校验模型路径：必须是允许目录下的 .pt 且文件存在。"""
    if not path:
        raise ValueError("未指定模型路径")
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    try:
        p = p.resolve()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法解析路径: {path}") from exc

    if p.suffix.lower() != ".pt":
        raise ValueError("只支持 .pt 模型文件")
    if not p.is_file():
        raise ValueError(f"模型文件不存在: {p}")

    roots = []
    for d in _model_dirs():
        try:
            roots.append(d.resolve())
        except Exception:  # noqa: BLE001
            continue
    if not any(p == r or r in p.parents for r in roots):
        raise ValueError("模型必须放在服务器的 weights/（或仓库根目录 / storage/models）下")

    return p


def list_local_models() -> list[dict]:
    """列出约定目录下的 .pt 文件（按修改时间倒序）。"""
    seen: set[Path] = set()
    items: list[dict] = []
    for d in allowed_dirs():
        for f in sorted(d.glob("*.pt")):
            try:
                rp = f.resolve()
                st = f.stat()
            except Exception:  # noqa: BLE001
                continue
            if rp in seen or not f.is_file():
                continue
            seen.add(rp)
            items.append({
                "path": str(rp),
                "name": f.name,
                "size_mb": round(st.st_size / 1024 / 1024, 1),
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items


def inspect_local_model(path: str | Path) -> dict:
    """加载 .pt 读取任务类型 / 类别数 / 是否 end2end（约 0.5~2s）。"""
    p = resolve_local_model(path)

    from ultralytics import YOLO

    model = YOLO(str(p))
    head = model.model.model[-1]

    names = getattr(model.model, "names", None) or {}
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    names = list(names)

    yaml_cfg = getattr(model.model, "yaml", None) or {}
    end2end = bool(getattr(head, "end2end", False)) or bool(yaml_cfg.get("end2end", False))

    return {
        "path": str(p),
        "name": p.name,
        "stem": p.stem,
        "task": model.task,
        "task_alias": TASK_ALIASES.get(model.task, model.task),
        "nc": int(getattr(head, "nc", len(names) or 0)),
        "names": names,
        "end2end": end2end,
    }
