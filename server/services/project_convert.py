# -*- coding: utf-8 -*-
"""
项目任务类型转换
================
把现有项目的 task_type 在 seg / det / obb / semseg 之间互转：
  - mode='inplace': 原地修改（保留同一个项目，只改 task_type）
  - mode='copy':    复制为新项目（保留原项目）

无论哪种模式，标注数据本身不需要修改：
  平台标注统一以 polygon JSON 存储。
  - seg 训练：用 polygon 生成 mask
  - det 训练：取 polygon 外接水平矩形作 bbox
  - obb 训练：取 polygon 最小外接旋转矩形作为 4 角点 OBB（cv2.minAreaRect）
  - semseg 训练：polygon 栅格化后**把所有类别实例合并成一张二值前景 mask**
    （见 ``dataset_service.prepare_semantic_dataset``），所以实例多边形是唯一的输入，
    转换时同样不需要改标注。

  注：seg ↔ obb 转换最有价值（多边形→旋转矩形保留角度信息）；
      det → obb 因为原本只有水平框，转后 OBB 全是 0°，等同 det。
      cls（图级标签）没有多边形，不参与任何转换。
"""

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from ..config import settings
from ..models.annotation import Annotation
from ..models.defect_class import DefectClass
from ..models.image import Image
from ..models.project import Project
from ..models.train_task import TrainTask

# 参与类型转换的任务类型（cls 是图级标签，没有多边形，不在此列）
POLYGON_TASK_TYPES = ("seg", "det", "obb", "semseg")


def _semseg_warnings(src: Project, db: Session) -> list[str]:
    """切换到 semseg 后的注意事项（不阻断转换，只提示）。

    语义分割与实例分割共用同一份实例多边形标注，差别全在训练端：
    ``prepare_semantic_dataset`` 把每图的全部实例栅格化后合并成一张二值前景 mask，
    类别索引只保留在 masks_ids 里（当前 num_classes=1，所有类别合并为“缺陷”）。
    """
    warnings = [
        "语义分割训练会把每张图的所有类别实例合并成一张二值前景 mask（0=背景 / 1=缺陷），"
        "不再区分实例和类别；标注数据本身不变，class_index 仍保留在 masks_ids 中供以后扩展。",
        "训练按缺陷中心采样 patch（默认 512），不预切图，也不使用项目里的 crop_size/overlap 滑窗参数。",
    ]
    labeled = (
        db.query(Image.id)
        .join(Annotation, Annotation.image_id == Image.id)
        .filter(Image.project_id == src.id)
        .distinct()
        .count()
    )
    if labeled == 0:
        warnings.append("当前项目还没有任何多边形标注，语义分割需要先标注缺陷（至少 2 张图才能开训）。")
    elif labeled < 2:
        warnings.append(f"当前只有 {labeled} 张图有缺陷标注，语义分割至少要 2 张才能开始训练。")
    return warnings


def _semseg_source_warnings(src: Project) -> list[str]:
    """从 semseg 转出时的提示：标注仍是实例多边形，可直接复用。"""
    return [
        "语义分割项目的标注同样是实例多边形（与 seg 完全一致），"
        "转回实例分割 / 检测 / OBB 不需要重新标注。",
    ]


def _resolve_project_name(db: Session, base_name: str) -> str:
    """项目重名时自动加后缀。"""
    existing = {p.name for p in db.query(Project.name).all()}
    if base_name not in existing:
        return base_name
    suffix = datetime.now().strftime("%Y%m%d")
    candidate = f"{base_name}_{suffix}"
    if candidate not in existing:
        return candidate
    i = 2
    while f"{candidate}_{i}" in existing:
        i += 1
    return f"{candidate}_{i}"


def convert_project_task_type(
    project_id: int,
    target_type: str,
    mode: str,
    db: Session,
    new_name: Optional[str] = None,
) -> dict:
    """
    转换项目任务类型（seg / det / obb / semseg）。

    Args:
        project_id: 源项目 ID
        target_type: 目标类型，'seg' / 'det' / 'obb' / 'semseg'
        mode: 'inplace' 或 'copy'
        db: SQLAlchemy Session
        new_name: copy 模式下的新项目名（可选，重名自动加后缀）

    Returns:
        {
            "mode": str,
            "source_project_id": int,
            "target_project_id": int,
            "target_project_name": str,
            "task_type": str,
            "image_count": int,
            "annotation_count": int,
            "warnings": list[str],   # 转换后的注意事项（如 semseg 会合并类别）
        }
    """
    if target_type not in POLYGON_TASK_TYPES:
        raise ValueError(
            f"target_type 必须是 {'/'.join(POLYGON_TASK_TYPES)}，got: {target_type}"
        )
    if mode not in ("inplace", "copy"):
        raise ValueError(f"mode 必须是 inplace 或 copy，got: {mode}")

    src = db.query(Project).filter(Project.id == project_id).first()
    if src is None:
        raise ValueError(f"项目 {project_id} 不存在")

    src_type = (src.task_type or "seg")
    if src_type == "cls":
        raise ValueError(
            "cls（图像分类）项目用的是图级标签 images.class_id，没有多边形标注，无法转换为 "
            f"{target_type}"
        )
    if src_type == target_type:
        raise ValueError(f"项目已经是 {target_type} 类型，无需转换")

    # ---- 安全检查：不允许在有活跃训练任务时原地转换 ----
    if mode == "inplace":
        active_states = ("pending", "preparing", "training", "exporting")
        active_task = (
            db.query(TrainTask)
            .filter(TrainTask.project_id == project_id, TrainTask.status.in_(active_states))
            .first()
        )
        if active_task:
            raise ValueError(
                f"项目存在活跃训练任务 (Task#{active_task.id}, status={active_task.status})，"
                f"请先取消后再原地转换。建议改用复制模式。"
            )

    if mode == "inplace":
        return _convert_inplace(src, target_type, db)
    return _convert_copy(src, target_type, new_name, db)


def _convert_inplace(src: Project, target_type: str, db: Session) -> dict:
    """原地修改 task_type。标注数据不动。"""
    src_type_before = (src.task_type or "seg")
    src.task_type = target_type
    # 检测项目通常 crop_size = 原图（不滑窗）；分割项目通常需要预处理
    # 这里不动其他字段，让用户自己在前端调整
    db.commit()

    img_count = db.query(Image).filter(Image.project_id == src.id).count()
    ann_count = (
        db.query(Annotation)
        .join(Image, Annotation.image_id == Image.id)
        .filter(Image.project_id == src.id)
        .count()
    )
    return {
        "mode": "inplace",
        "source_project_id": src.id,
        "target_project_id": src.id,
        "target_project_name": src.name,
        "task_type": target_type,
        "image_count": img_count,
        "annotation_count": ann_count,
        "warnings": (
            _semseg_warnings(src, db) if target_type == "semseg"
            else _semseg_source_warnings(src) if src_type_before == "semseg"
            else []
        ),
    }


def _convert_copy(
    src: Project, target_type: str, new_name: Optional[str], db: Session
) -> dict:
    """复制项目为新 task_type。复制图片文件、类别、标注。"""
    src_type_before = (src.task_type or "seg")
    base_name = (new_name or "").strip() or f"{src.name}_{target_type}"
    final_name = _resolve_project_name(db, base_name)

    new_project = Project(
        name=final_name,
        description=(src.description or "") + f" (从 #{src.id} 转换为 {target_type})",
        task_type=target_type,
        resize_h=src.resize_h,
        resize_w=src.resize_w,
        crop_size=src.crop_size,
        overlap=src.overlap,
    )
    db.add(new_project)
    db.flush()

    # 复制类别（class_index 不变）
    src_classes = db.query(DefectClass).filter(DefectClass.project_id == src.id).all()
    class_id_map: dict[int, int] = {}  # 老 class_id → 新 class_id
    for dc in src_classes:
        new_dc = DefectClass(
            project_id=new_project.id,
            class_index=dc.class_index,
            name=dc.name,
            color=dc.color,
        )
        db.add(new_dc)
        db.flush()
        class_id_map[dc.id] = new_dc.id

    # 复制图片文件 + Image 记录
    src_upload = settings.upload_path / str(src.id)
    new_upload = settings.upload_path / str(new_project.id)
    new_upload.mkdir(parents=True, exist_ok=True)

    src_images = db.query(Image).filter(Image.project_id == src.id).all()
    image_id_map: dict[int, int] = {}
    img_count = 0
    _hex_chars = set("0123456789abcdef")
    for img in src_images:
        # 物理文件路径（兼容相对路径与绝对路径）
        src_fp = Path(img.file_path)
        if not src_fp.is_absolute():
            src_fp = settings.upload_path / img.file_path
        if not src_fp.exists():
            continue

        # 新文件名带新 uuid，避免冲突
        original_basename = Path(img.file_path).name
        # 移除原 uuid 前缀（如果有），保留可读名。
        # 必须同时校验前缀是 8 位 *十六进制*，否则用户上传名 "abcdefgh_xxx.png"
        # 会被误剥成 "xxx.png"。
        if "_" in original_basename:
            prefix = original_basename.split("_", 1)[0]
            if len(prefix) == 8 and set(prefix.lower()).issubset(_hex_chars):
                tail = original_basename.split("_", 1)[1]
            else:
                tail = original_basename
        else:
            tail = original_basename
        new_basename = f"{uuid.uuid4().hex[:8]}_{tail}"
        new_fp = new_upload / new_basename
        shutil.copy2(str(src_fp), str(new_fp))

        new_img = Image(
            project_id=new_project.id,
            filename=img.filename,
            source_relative_path=img.source_relative_path,
            file_path=f"{new_project.id}/{new_basename}",
            width=img.width,
            height=img.height,
            file_size=img.file_size,
            # 文件是逐字节 copy2，内容哈希与原图一致，可直接沿用
            content_hash=img.content_hash,
            status=img.status,
            annotator=img.annotator,
            reviewer=img.reviewer,
        )
        db.add(new_img)
        db.flush()
        image_id_map[img.id] = new_img.id
        img_count += 1

    # 复制标注
    src_anns = (
        db.query(Annotation)
        .filter(Annotation.image_id.in_(image_id_map.keys()))
        .all()
    ) if image_id_map else []
    ann_count = 0
    for ann in src_anns:
        new_image_id = image_id_map.get(ann.image_id)
        new_class_id = class_id_map.get(ann.class_id)
        if new_image_id is None or new_class_id is None:
            continue
        new_ann = Annotation(
            image_id=new_image_id,
            class_id=new_class_id,
            polygon=ann.polygon,
            # 矩形/圆形保持参数化（polygon 仍是几何真相，shape 只是编辑提示）
            shape=ann.shape or "polygon",
            shape_params=ann.shape_params,
            area=ann.area,
            bbox=ann.bbox,
            created_by=ann.created_by,
        )
        db.add(new_ann)
        ann_count += 1

    db.commit()

    return {
        "mode": "copy",
        "source_project_id": src.id,
        "target_project_id": new_project.id,
        "target_project_name": new_project.name,
        "task_type": target_type,
        "image_count": img_count,
        "annotation_count": ann_count,
        "warnings": (
            _semseg_warnings(new_project, db) if target_type == "semseg"
            else _semseg_source_warnings(src) if src_type_before == "semseg"
            else []
        ),
    }
