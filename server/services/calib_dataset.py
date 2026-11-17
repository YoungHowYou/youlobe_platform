# -*- coding: utf-8 -*-
"""INT8 校准数据集解析（OpenVINO / ultralytics PTQ）。

背景：训练结束（completed / cancelled）后 ``purge_task_intermediates`` 会删掉
``dataset/``（可再生的大体积中间产物），但 ultralytics 的 INT8 导出需要
``data=<dataset.yaml>`` 做量化校准。历史实现把 ``<task>/dataset`` 直接塞给导出，
任务一归档就必然报 ``…/dataset does not exist``。

解析顺序：

1. ``<task>/dataset``：训练产物还在，口径最准（同一批切片）
2. ``<task>/calib``：训练收尾时抽样保留的 mini 校准集（清理/归档后仍可用）
3. 用项目原图 + 标注现建 ``<task>/calib``：老任务兜底（不切片，直接用原图）

``calib/`` 的布局与 YOLO 数据集一致（``images/`` + ``labels/`` + ``dataset.yaml``），
分类任务则是 ImageFolder（``train/<cls>/`` + ``val/<cls>/``）。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CALIB_LIMIT = 64          # 默认抽样张数（NNCF 校准几十张足够，且不占空间）
CALIB_DIRNAME = "calib"


def _list_images(root: Path) -> list[Path]:
    if not root or not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _label_for(img: Path) -> Optional[Path]:
    """``…/images/<split>/x.png`` → ``…/labels/<split>/x.txt``。"""
    parts = list(img.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    cand = img.with_suffix(".txt")
    return cand if cand.exists() else None


def write_dataset_yaml(out_dir: Path, class_names: list[str] | None, *,
                       cls: bool = False) -> Path:
    """写 ``dataset.yaml``；分类任务用 ImageFolder 的 train/val 子目录。"""
    names = [str(n) for n in (class_names or []) if str(n).strip()] or ["defect"]
    lines = [
        f"path: {out_dir}",
        f"train: {'train' if cls else 'images'}",
        f"val: {'val' if cls else 'images'}",
        "names:",
    ]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    yml = out_dir / "dataset.yaml"
    yml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yml


# ---------------------------------------------------------------- 从训练产物抽样
def make_calib_from_dataset(dataset_dir: str | Path, task_type: str,
                            class_names: list[str] | None,
                            out_dir: str | Path,
                            limit: int = CALIB_LIMIT) -> Optional[Path]:
    """从训练数据集抽样一份 mini 校准集，返回 ``out_dir``；没图时返回 None。"""
    src = Path(dataset_dir)
    out = Path(out_dir)
    if not src.is_dir():
        return None
    if task_type == "cls":
        return _make_cls_calib(src, out, class_names, limit)

    imgs = _list_images(src)
    if not imgs:
        return None
    out_img, out_lbl = out / "images", out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    kept = 0
    for img in imgs[:max(1, int(limit))]:
        stem = f"{kept:04d}"
        shutil.copy2(img, out_img / f"{stem}{img.suffix.lower()}")
        lbl = _label_for(img)
        if lbl and lbl.is_file():
            shutil.copy2(lbl, out_lbl / f"{stem}.txt")
        else:
            (out_lbl / f"{stem}.txt").write_text("", encoding="utf-8")
        kept += 1
    if kept == 0:
        return None
    write_dataset_yaml(out, class_names)
    return out


def _make_cls_calib(src: Path, out: Path, class_names: list[str] | None,
                    limit: int) -> Optional[Path]:
    """分类数据集（ImageFolder ``<class>/img``）抽样。"""
    subs = [p for p in sorted(src.iterdir()) if p.is_dir()]
    # 兼容 ultralytics 风格的 <dataset>/train/<cls>
    if subs and {p.name for p in subs} & {"train", "val"}:
        subs = [p for p in sorted((src / "train").iterdir()) if p.is_dir()] \
            if (src / "train").is_dir() else []
    if not subs:
        return None
    per = max(4, int(limit) // max(1, len(subs)))
    copied = 0
    for sub in subs:
        for split in ("train", "val"):
            (out / split / sub.name).mkdir(parents=True, exist_ok=True)
        for f in _list_images(sub)[:per]:
            shutil.copy2(f, out / "train" / sub.name / f.name)
            shutil.copy2(f, out / "val" / sub.name / f.name)
            copied += 1
    if copied == 0:
        return None
    write_dataset_yaml(out, class_names or [p.name for p in subs], cls=True)
    return out


# ---------------------------------------------------------------- 从项目原图现建
def _image_size(img, upload_root: Path) -> tuple[int, int]:
    w, h = int(getattr(img, "width", 0) or 0), int(getattr(img, "height", 0) or 0)
    if w > 0 and h > 0:
        return w, h
    try:
        from PIL import Image as PILImage
        with PILImage.open(upload_root / img.file_path) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return 0, 0


def _polygon_points(poly) -> list[dict]:
    """标注 polygon 可能是 ``[{"x","y"}, …]`` 或 ``[[x,y], …]``。"""
    pts = []
    for p in poly or []:
        if isinstance(p, dict):
            pts.append({"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0))})
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append({"x": float(p[0]), "y": float(p[1])})
    return pts


def _label_line(task_type: str, cls_idx: int, pts: list[dict],
                w: int, h: int) -> str | None:
    if len(pts) < 3:
        return None
    if task_type == "seg":
        return f"{cls_idx} " + " ".join(
            f"{min(max(p['x'], 0.0), 1.0):.6f} {min(max(p['y'], 0.0), 1.0):.6f}"
            for p in pts
        )
    if task_type == "det":
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        x1, x2 = min(max(min(xs), 0.0), 1.0), min(max(max(xs), 0.0), 1.0)
        y1, y2 = min(max(min(ys), 0.0), 1.0), min(max(max(ys), 0.0), 1.0)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return None
        return f"{cls_idx} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {bw:.6f} {bh:.6f}"
    if task_type == "obb":
        if w <= 0 or h <= 0:
            return None
        try:
            import cv2
            import numpy as np
            arr = np.array([[p["x"] * w, p["y"] * h] for p in pts], dtype=np.float32)
            (cx, cy), (bw, bh), ang = cv2.minAreaRect(arr)
            box = cv2.boxPoints(((cx, cy), (bw, bh), ang))
            coords = []
            for x, y in box:
                coords += [min(max(float(x) / w, 0.0), 1.0),
                           min(max(float(y) / h, 0.0), 1.0)]
            return f"{cls_idx} " + " ".join(f"{c:.6f}" for c in coords)
        except Exception:  # noqa: BLE001
            return None
    return None


def build_calib_from_project(project, db, out_dir: str | Path,
                             limit: int = CALIB_LIMIT) -> Optional[Path]:
    """用项目原图 + 标注现建校准集（训练产物已被清理时的兜底）。"""
    from ..config import settings
    from ..models.annotation import Annotation
    from ..models.defect_class import DefectClass
    from ..models.image import Image

    if project is None:
        return None
    out = Path(out_dir)
    task_type = (project.task_type or "seg").lower()
    classes = (
        db.query(DefectClass)
        .filter(DefectClass.project_id == project.id)
        .order_by(DefectClass.class_index)
        .all()
    )
    class_names = [c.name or f"class_{c.class_index}" for c in classes]
    idx_by_id = {c.id: int(c.class_index) for c in classes}
    upload_root = Path(settings.upload_path)

    if task_type == "cls":
        return _build_cls_from_project(project, db, out, upload_root, limit)

    rows = (
        db.query(Image)
        .join(Annotation, Annotation.image_id == Image.id)
        .filter(Image.project_id == project.id)
        .distinct()
        .limit(max(1, int(limit)) * 2)
        .all()
    )
    out_img, out_lbl = out / "images", out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    kept = 0
    for img in rows:
        if kept >= limit:
            break
        src = upload_root / img.file_path
        if not src.is_file():
            continue
        lines = []
        for ann in img.annotations:
            cls_idx = idx_by_id.get(ann.class_id)
            if cls_idx is None:
                continue
            w, h = _image_size(img, upload_root)
            line = _label_line(task_type, cls_idx, _polygon_points(ann.polygon), w, h)
            if line:
                lines.append(line)
        stem = f"{kept:04d}"
        shutil.copy2(src, out_img / f"{stem}{src.suffix.lower() or '.png'}")
        (out_lbl / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        kept += 1
    if kept == 0:
        return None
    write_dataset_yaml(out, class_names)
    return out


def _build_cls_from_project(project, db, out: Path, upload_root: Path,
                            limit: int) -> Optional[Path]:
    from ..models.defect_class import DefectClass
    from ..models.image import Image

    classes = (
        db.query(DefectClass)
        .filter(DefectClass.project_id == project.id)
        .order_by(DefectClass.class_index)
        .all()
    )
    idx_by_id = {c.id: int(c.class_index) for c in classes}
    if not idx_by_id:
        return None
    per = max(2, int(limit) // max(1, len(idx_by_id)))
    rows = (
        db.query(Image)
        .filter(Image.project_id == project.id, Image.class_id.isnot(None))
        .all()
    )
    copied = 0
    for img in rows:
        cls_idx = idx_by_id.get(img.class_id)
        if cls_idx is None:
            continue
        src = upload_root / img.file_path
        if not src.is_file():
            continue
        folder = f"{cls_idx:04d}"
        for split in ("train", "val"):
            dst = out / split / folder
            dst.mkdir(parents=True, exist_ok=True)
        trainers = list((out / "train" / folder).glob(f"{img.id}.*"))
        if len(trainers) >= per:
            continue
        shutil.copy2(src, out / "train" / folder / f"{img.id}{src.suffix.lower()}")
        shutil.copy2(src, out / "val" / folder / f"{img.id}{src.suffix.lower()}")
        copied += 1
    if copied == 0:
        return None
    write_dataset_yaml(out, [c.name or f"class_{c.class_index}" for c in classes], cls=True)
    return out


# ---------------------------------------------------------------- 对外入口
def keep_calib_for_task(task, db, limit: int = CALIB_LIMIT) -> Optional[Path]:
    """训练收尾（清理中间产物之前）抽样保留 mini 校准集，供后续 INT8 导出。"""
    out_root = Path(task.output_dir) if getattr(task, "output_dir", None) else None
    if out_root is None:
        return None
    calib = out_root / CALIB_DIRNAME
    if (calib / "dataset.yaml").is_file():
        return calib
    dataset = out_root / "dataset"
    if not dataset.is_dir():
        return None
    project = None
    try:
        from ..models.project import Project
        project = db.query(Project).filter(Project.id == task.project_id).first()
    except Exception:  # noqa: BLE001
        project = None
    task_type = (project.task_type if project else "seg") or "seg"
    class_names = list((task.config or {}).get("class_names") or [])
    made = make_calib_from_dataset(dataset, task_type, class_names, calib, limit)
    if made:
        print(f"[训练任务] Task#{task.id} 已保留 INT8 校准集：{made}")
    return made


def ensure_calib_dataset(task, project, db,
                         limit: int = CALIB_LIMIT) -> tuple[Optional[str], str]:
    """解析 INT8 校准数据集，返回 ``(含 dataset.yaml 的目录, 说明)``。"""
    out_root = Path(task.output_dir) if getattr(task, "output_dir", None) else None
    if out_root is not None:
        dataset = out_root / "dataset"
        if any(dataset.glob("*.yaml")):
            return str(dataset), "使用训练数据集校准"
        calib = out_root / CALIB_DIRNAME
        if any(calib.glob("*.yaml")):
            return str(calib), "使用训练收尾时保留的校准集"
    if out_root is None or project is None:
        return None, "任务目录或项目信息缺失，无法构建校准集"
    try:
        made = build_calib_from_project(project, db, out_root / CALIB_DIRNAME, limit)
    except Exception as exc:  # noqa: BLE001
        return None, f"从项目原图重建校准集失败：{exc}"
    if made is None:
        return None, "项目里没有可用的图片/标注，无法构建校准集"
    return str(made), "训练中间产物已清理，已从项目原图重建校准集"
