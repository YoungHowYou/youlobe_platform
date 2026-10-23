# -*- coding: utf-8 -*-
"""
删除项目 / 训练任务时的磁盘产物清理（唯一收口）
================================================
平台会产生这些磁盘产物：

    storage/uploads/<project_id>/                原图 + thumbs/        ← 项目级
    storage/runs/task_<task_id>/                 训练全套中间产物      ← 任务级
        raw/            数据库导出的原图 + YOLO txt
        preprocessed/   形态学三通道产物
        cropped/        滑窗切图（训练实际吃的）
        dataset/        images/labels + dataset.yaml
        runs/train/     weights / results.csv / args.yaml / plots / labels.cache
        runs/train/exports/   ONNX / OpenVINO 导出（OpenVINO 是目录）
    storage/runs/inference/<id>_*.png            推理可视化图          ← 任务级
    <tmp>/model_downloads/<name>_id<record>.zip  导出模型下载缓存      ← 任务级

「删除项目 / 删除训练任务」接口都要通过本模块清理，避免各处各写一份
shutil.rmtree 逻辑、漏掉某类产物。所有删除都限制在 STORAGE_ROOT（或系统临时
目录下的 model_downloads）内，防止 DB 被改后误删其它文件。
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from ..config import settings

# 下载缓存目录（与 server/routers/export.py 保持一致）
DOWNLOAD_CACHE_DIR = Path(tempfile.gettempdir()) / "model_downloads"

_STATIC_PREFIX = "/static/storage/"


def storage_root() -> Path:
    return Path(settings.STORAGE_ROOT).resolve()


def runs_root() -> Path:
    return (storage_root() / settings.RUNS_DIR).resolve()


def uploads_root() -> Path:
    return (storage_root() / settings.UPLOAD_DIR).resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root)
    except Exception:  # noqa: BLE001 - 路径异常一律当作“不在范围内”
        return False


def remove_tree(path: Optional[Path], root: Path) -> bool:
    """删除 root 内的目录；返回是否真的删掉了。"""
    if path is None:
        return False
    try:
        if not path.exists() or not _inside(path, root):
            return False
        shutil.rmtree(str(path), ignore_errors=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def remove_path(path: Optional[Path | str], root: Path) -> bool:
    """删除 root 内的文件或目录（导出可能是一目录）。"""
    if not path:
        return False
    try:
        p = Path(path)
        if not p.exists() or not _inside(p, root):
            return False
        if p.is_dir():
            shutil.rmtree(str(p), ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def task_dir(task_id: int, output_dir: Optional[str] = None) -> Path:
    """任务的产物根目录：优先用记录里的 output_dir，否则回退 runs/task_<id>。"""
    if output_dir:
        return Path(output_dir)
    return runs_root() / f"task_{task_id}"


def url_to_disk(url: str) -> Optional[Path]:
    """把 /static/storage/... 反解成磁盘路径（限制在 STORAGE_ROOT 内）。"""
    if not url:
        return None
    rel = url.replace(_STATIC_PREFIX, "", 1).lstrip("/\\")
    if not rel:
        return None
    target = (storage_root() / rel).resolve()
    return target if _inside(target, storage_root()) else None


def remove_download_zip_cache(record_ids: Iterable[int]) -> int:
    """清掉导出模型下载缓存的 zip（文件名带 `_id<ExportedModel.id>`）。"""
    removed = 0
    ids = [int(i) for i in record_ids if i is not None]
    if not ids or not DOWNLOAD_CACHE_DIR.is_dir():
        return 0
    for rid in ids:
        for path in DOWNLOAD_CACHE_DIR.glob(f"*_id{rid}.zip"):
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except Exception:  # noqa: BLE001
                pass
    return removed


def remove_inference_files(urls: Iterable[str]) -> int:
    removed = 0
    for url in urls:
        target = url_to_disk(url)
        if target is None:
            continue
        try:
            if target.exists():
                target.unlink(missing_ok=True)
                removed += 1
        except Exception:  # noqa: BLE001
            pass
    # 推理图都平铺在 runs/inference/ 下，空了就把目录本身也收掉
    infer_dir = runs_root() / "inference"
    try:
        if infer_dir.is_dir() and not any(infer_dir.iterdir()):
            infer_dir.rmdir()
    except Exception:  # noqa: BLE001
        pass
    return removed


def sweep_orphan_task_dirs(valid_task_ids: Iterable[int]) -> list[str]:
    """删掉 runs/task_<n> 里 DB 中已不存在的任务目录（历史删除失败的残留）。"""
    valid = {int(i) for i in valid_task_ids}
    removed: list[str] = []
    root = runs_root()
    if not root.is_dir():
        return removed
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("task_"):
            continue
        try:
            tid = int(child.name[len("task_"):])
        except ValueError:
            continue                      # task_xxx 之类的非数字目录不动
        if tid in valid:
            continue
        if remove_tree(child, root):
            removed.append(child.name)
    return removed


def sweep_orphan_upload_dirs(valid_project_ids: Iterable[int]) -> list[str]:
    """删掉 uploads/<pid> 里 DB 中已不存在的项目目录。"""
    valid = {int(i) for i in valid_project_ids}
    removed: list[str] = []
    root = uploads_root()
    if not root.is_dir():
        return removed
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            pid = int(child.name)
        except ValueError:
            continue
        if pid in valid:
            continue
        if remove_tree(child, root):
            removed.append(child.name)
    return removed


# 训练后才需要、训练结束就能删的中间产物（都能由 DB 标注 + uploads 重新生成）
TRANSIENT_DIRS = ("raw", "preprocessed", "cropped", "dataset")
TRANSIENT_FILES = ("dataset.yaml",)


def purge_task_intermediates(
    task_id: int,
    output_dir: Optional[str] = None,
    keep_previews: bool = True,
) -> dict:
    """删掉一个任务训练完成后不再需要的中间产物，返回释放的空间。

    删：raw/ preprocessed/ cropped/ dataset/ dataset.yaml
        runs/*/weights/epoch*.pt（周期性快照，动辄几百 MB）
        runs/*/labels.cache
    保：weights/best.pt、last.pt（继承训练 / 导出 / 在线推理）
        results.csv（训练曲线回填）、args.yaml（超参记录）
        results.png / confusion_matrix / custom_plots 等可视化
        exports/（已导出的 ONNX/OpenVINO）
        train_batch*.jpg / val_batch*.jpg（标注预览，keep_previews=False 时才删）
    """
    base = task_dir(task_id, output_dir)
    result = {"task_id": task_id, "freed_mb": 0.0, "removed": [], "kept_root": str(base)}
    if not base.exists():
        return result

    def _unlink(p: Path) -> None:
        nonlocal freed
        try:
            size = p.stat().st_size
            p.unlink(missing_ok=True)
            freed += size / 1024 / 1024
            result["removed"].append(str(p.relative_to(base)))
        except Exception:  # noqa: BLE001
            pass

    freed = 0.0
    for name in TRANSIENT_DIRS:
        d = base / name
        if d.is_dir() and _inside(d, base):
            before = dir_size_mb(d)
            if remove_tree(d, base):
                freed += before
                result["removed"].append(name + "/")
    for name in TRANSIENT_FILES:
        f = base / name
        if f.is_file():
            _unlink(f)

    runs_dir = base / "runs"
    if runs_dir.is_dir():
        for run_root in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            weights = run_root / "weights"
            if weights.is_dir():
                for ckpt in sorted(weights.glob("epoch*.pt")):
                    _unlink(ckpt)
            for cache in run_root.glob("labels.cache"):
                _unlink(cache)
            if not keep_previews:
                for pat in ("train_batch*.jpg", "val_batch*.jpg"):
                    for img in run_root.glob(pat):
                        _unlink(img)

    result["freed_mb"] = round(freed, 1)
    return result


def dir_size_mb(path: Path) -> float:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except Exception:  # noqa: BLE001
        return 0.0
    return round(total / 1024 / 1024, 1)
