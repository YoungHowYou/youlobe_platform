# -*- coding: utf-8 -*-
"""按缺陷尺寸分桶评估实例分割效果（离线指标工具）。

背景：Ultralytics 的实例掩膜损失按目标框面积归一，小目标本来就被放大；
继续在损失里叠加小实例权重容易过度放大噪声。要看小缺陷是否真的变好，
应该直接按尺寸分桶统计召回 / IoU，而不是改损失。

用法：
    python tools/eval_size_buckets.py --model storage/runs/task_12/runs/train/weights/best.pt \
        --data storage/runs/task_12/dataset/dataset.yaml

    # 只看前 100 张、按掩膜 IoU 匹配、输出 JSON
    python tools/eval_size_buckets.py --model best.pt --data dataset.yaml \
        --limit 100 --mask-iou --json buckets.json

输出：按 GT 框最大边（像素）分桶的召回、平均 IoU，以及按预测尺寸分桶的误检数。
本工具只做离线评估，不参与训练、不改任何权重/配置。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def _resolve_split(data_yaml: Path, split_key: str) -> Path:
    """读取 dataset.yaml，返回指定 split 的绝对路径（目录或图片列表 txt）。"""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    split = cfg.get(split_key)
    if not split:
        raise SystemExit(f"{data_yaml} 里没有 '{split_key}' 字段")

    root = Path(cfg.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    target = Path(split)
    if target.is_absolute():
        return target
    candidate = root / target
    return candidate if candidate.exists() else (data_yaml.parent / target).resolve()


def _collect_images(split_path: Path) -> List[Path]:
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        lines = split_path.read_text(encoding="utf-8").splitlines()
        base = split_path.parent
        images = []
        for line in lines:
            item = line.strip()
            if not item:
                continue
            path = Path(item)
            images.append(path if path.is_absolute() else (base / path).resolve())
        return [p for p in images if p.exists()]

    if not split_path.is_dir():
        raise SystemExit(f"找不到验证集路径: {split_path}")
    return sorted(p for p in split_path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _guess_label_dir(image: Path, split_path: Path) -> Path:
    """按 YOLO 约定从 images 目录推 labels 目录。"""
    parts = list(split_path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            parts[idx] = "labels"
            return Path(*parts)
    return split_path.parent / "labels"


def _read_gt_labels(label_path: Path, img_w: int, img_h: int) -> List[Dict]:
    """解析 YOLO 标注：支持 seg 多边形与 det/obb 框。"""
    if not label_path.exists():
        return []

    items: List[Dict] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls_id = int(float(values[0]))
        coords = [float(v) for v in values[1:]]

        if len(coords) == 4:  # det/obb: cx cy w h（obb 的额外角度忽略）
            cx, cy, bw, bh = coords
            x1, y1 = (cx - bw / 2) * img_w, (cy - bh / 2) * img_h
            x2, y2 = (cx + bw / 2) * img_w, (cy + bh / 2) * img_h
            polygon = None
        elif len(coords) >= 6 and len(coords) % 2 == 0:  # seg 多边形
            points = [(coords[i] * img_w, coords[i + 1] * img_h) for i in range(0, len(coords), 2)]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            polygon = points
        else:
            continue

        items.append({
            "cls": cls_id,
            "bbox": (x1, y1, x2, y2),
            "size": max(x2 - x1, y2 - y1),
            "polygon": polygon,
        })
    return items


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _polygon_to_mask(polygon, height: int, width: int):
    import numpy as np
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _mask_iou(gt_mask, pred_mask) -> float:
    import numpy as np

    gt = np.asarray(gt_mask, dtype=bool)
    pred = np.asarray(pred_mask, dtype=bool)
    if gt.shape != pred.shape:
        return 0.0
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(inter) / float(union) if union else 0.0


def _bucket_edges(raw: str) -> List[float]:
    edges = [float(v) for v in raw.split(",") if v.strip()]
    if not edges or edges != sorted(edges):
        raise SystemExit("--buckets 需要升序数值，例如 16,32,64")
    return edges


def _bucket_name(size: float, edges: List[float]) -> str:
    low = 0.0
    for edge in edges:
        if size < edge:
            return f"<{edge:g}px" if low == 0 else f"{low:g}-{edge:g}px"
        low = edge
    return f">={edges[-1]:g}px"


def _bucket_names(edges: List[float]) -> List[str]:
    names: List[str] = []
    low = 0.0
    for edge in edges:
        names.append(f"<{edge:g}px" if low == 0 else f"{low:g}-{edge:g}px")
        low = edge
    names.append(f">={edges[-1]:g}px")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="按缺陷尺寸分桶评估实例分割召回/IoU")
    parser.add_argument("--model", required=True, help="权重路径（.pt）")
    parser.add_argument("--data", required=True, help="dataset.yaml 路径")
    parser.add_argument("--split", default="val", help="数据集 split（默认 val）")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸（默认 640）")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值（默认 0.001，尽量多召回）")
    parser.add_argument("--iou", type=float, default=0.5, help="匹配用 IoU 阈值（默认 0.5）")
    parser.add_argument("--buckets", default="16,32,64", help="尺寸分桶边界，单位为像素最大边")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 张图（0 = 全部）")
    parser.add_argument("--mask-iou", action="store_true", help="匹配时使用掩膜 IoU（默认用框 IoU）")
    parser.add_argument("--device", default=None, help="推理设备，例如 0 / cpu")
    parser.add_argument("--json", dest="json_out", default=None, help="结果另存为 JSON")
    args = parser.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.exists():
        raise SystemExit(f"dataset.yaml 不存在: {data_yaml}")

    split_path = _resolve_split(data_yaml, args.split)
    images = _collect_images(split_path)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"{split_path} 下没有找到图片")

    edges = _bucket_edges(args.buckets)
    bucket_names = _bucket_names(edges)
    per_bucket = {
        name: {"gt": 0, "matched": 0, "box_iou_sum": 0.0, "mask_iou_sum": 0.0, "mask_iou_n": 0}
        for name in bucket_names
    }
    pred_bucket = {name: {"pred": 0, "false_positive": 0} for name in bucket_names}

    import cv2
    from ultralytics import YOLO

    model = YOLO(str(Path(args.model).expanduser().resolve()))
    predict_kwargs = {"imgsz": args.imgsz, "conf": args.conf, "iou": 0.7, "verbose": False}
    if args.device:
        predict_kwargs["device"] = args.device

    label_dir = _guess_label_dir(images[0], split_path)
    total_gt = total_matched = 0

    for index, image_path in enumerate(images, start=1):
        frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            continue
        img_h, img_w = frame.shape[:2]

        label_path = label_dir / f"{image_path.stem}.txt"
        gts = _read_gt_labels(label_path, img_w, img_h)

        result = model.predict(source=str(image_path), **predict_kwargs)[0]
        boxes = result.boxes
        pred_boxes: List[Tuple[float, float, float, float]] = []
        pred_cls: List[int] = []
        pred_masks = None
        if boxes is not None and len(boxes) > 0:
            pred_boxes = [tuple(float(v) for v in xyxy) for xyxy in boxes.xyxy.tolist()]
            pred_cls = [int(v) for v in boxes.cls.tolist()]
            if getattr(result, "masks", None) is not None and result.masks is not None:
                pred_masks = result.masks.data.cpu().numpy()

        # 贪心匹配：同类、IoU 高于阈值、一对一
        candidates: List[Tuple[float, int, int]] = []
        for gi, gt in enumerate(gts):
            for pi, pbox in enumerate(pred_boxes):
                if pred_cls[pi] != gt["cls"]:
                    continue
                if args.mask_iou and pred_masks is not None and gt["polygon"]:
                    gt_mask = _polygon_to_mask(gt["polygon"], img_h, img_w)
                    iou = _mask_iou(gt_mask, pred_masks[pi] > 0.5)
                else:
                    iou = _box_iou(gt["bbox"], pbox)
                if iou >= args.iou:
                    candidates.append((iou, gi, pi))

        candidates.sort(reverse=True)
        matched_gt, matched_pred = set(), set()
        for iou, gi, pi in candidates:
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)
            name = _bucket_name(gts[gi]["size"], edges)
            per_bucket[name]["matched"] += 1
            per_bucket[name]["box_iou_sum"] += _box_iou(gts[gi]["bbox"], pred_boxes[pi])
            if pred_masks is not None and gts[gi]["polygon"]:
                gt_mask = _polygon_to_mask(gts[gi]["polygon"], img_h, img_w)
                per_bucket[name]["mask_iou_sum"] += _mask_iou(gt_mask, pred_masks[pi] > 0.5)
                per_bucket[name]["mask_iou_n"] += 1

        for gt in gts:
            per_bucket[_bucket_name(gt["size"], edges)]["gt"] += 1
        total_gt += len(gts)
        total_matched += len(matched_gt)

        for pi, pbox in enumerate(pred_boxes):
            psize = max(pbox[2] - pbox[0], pbox[3] - pbox[1])
            entry = pred_bucket[_bucket_name(psize, edges)]
            entry["pred"] += 1
            if pi not in matched_pred:
                entry["false_positive"] += 1

        if index % 50 == 0:
            print(f"[size buckets] 已处理 {index}/{len(images)} 张")

    print()
    print(f"模型: {args.model}")
    print(f"验证集: {split_path}（{len(images)} 张，匹配阈值 IoU={args.iou}，conf={args.conf}）")
    print()
    header = f"{'尺寸桶':<14}{'GT':>6}{'命中':>6}{'召回':>9}{'框IoU':>9}{'掩膜IoU':>10}{'预测':>7}{'误检':>7}"
    print(header)
    print("-" * len(header))
    for name in bucket_names:
        stat = per_bucket[name]
        pred_stat = pred_bucket[name]
        recall = stat["matched"] / stat["gt"] if stat["gt"] else 0.0
        box_iou = stat["box_iou_sum"] / stat["matched"] if stat["matched"] else 0.0
        mask_iou = (stat["mask_iou_sum"] / stat["mask_iou_n"]) if stat["mask_iou_n"] else 0.0
        print(
            f"{name:<14}{stat['gt']:>6}{stat['matched']:>6}{recall:>9.3f}"
            f"{box_iou:>9.3f}{mask_iou:>10.3f}{pred_stat['pred']:>7}{pred_stat['false_positive']:>7}"
        )

    overall_recall = total_matched / total_gt if total_gt else 0.0
    print("-" * len(header))
    print(f"{'合计':<14}{total_gt:>6}{total_matched:>6}{overall_recall:>9.3f}")

    if args.json_out:
        payload = {
            "model": str(args.model),
            "data": str(data_yaml),
            "split": args.split,
            "images": len(images),
            "iou_threshold": args.iou,
            "conf": args.conf,
            "buckets": {name: {**per_bucket[name], **pred_bucket[name]} for name in bucket_names},
            "overall": {"gt": total_gt, "matched": total_matched, "recall": overall_recall},
        }
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已写入 {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
