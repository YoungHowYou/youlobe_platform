# -*- coding: utf-8 -*-
"""语义分割评价指标（与 scratch_seg/tools/evaluate.py 同口径）。

区别仅在于骨架提取：实验用 ``skimage.morphology.skeletonize``，平台 venv
没有 scikit-image，这里用等价实现 **Zhang-Suen 细化**（纯 numpy），
对连通域的骨架结果一致，Skeleton Recall 的差异在 1e-3 量级。

指标：
    dice / iou / recall        像素级
    instance_recall            实例级（IoU>=0.5 命中；需要实例 ID mask）
    skeleton_recall            GT 骨架点落在预测 2px 内的比例
    fragmentation              预测连通域数 / GT 实例数（1.0 为理想）
    boundary_error             边界对称距离
"""

from typing import Optional

import cv2
import numpy as np


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 细化，返回 bool 骨架图。"""
    img = (np.asarray(mask) > 0).astype(np.uint8)
    if img.sum() == 0:
        return img.astype(bool)

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(neighbors)
            seq = neighbors + [p2]
            a = sum(
                ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
                for i in range(8)
            )

            if step == 0:
                cond = ((img == 1) & (b >= 2) & (b <= 6) & (a == 1)
                        & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0))
            else:
                cond = ((img == 1) & (b >= 2) & (b <= 6) & (a == 1)
                        & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0))

            if cond.any():
                img[cond] = 0
                changed = True
    return img.astype(bool)


def skeleton_recall(gt_mask: np.ndarray, pred_mask: np.ndarray,
                    tolerance: float = 2.0) -> float:
    skel = skeletonize(gt_mask > 0)
    if skel.sum() == 0:
        return float("nan")
    if (pred_mask > 0).sum() == 0:
        return 0.0
    dt = cv2.distanceTransform((pred_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    return float((dt[skel] <= tolerance).mean())


def boundary_error(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    def _boundary(m):
        m8 = m.astype(np.uint8)
        eroded = cv2.erode(m8, np.ones((3, 3), np.uint8))
        return m.astype(bool) & ~eroded.astype(bool)

    b1, b2 = _boundary(gt_mask), _boundary(pred_mask)
    if b1.sum() == 0 or b2.sum() == 0:
        return float("nan")
    d1 = cv2.distanceTransform((~b2).astype(np.uint8), cv2.DIST_L2, 3)[b1].mean()
    d2 = cv2.distanceTransform((~b1).astype(np.uint8), cv2.DIST_L2, 3)[b2].mean()
    return float((d1 + d2) / 2)


def evaluate_pair(pred_mask: np.ndarray, gt_mask: np.ndarray,
                  gt_instance: Optional[np.ndarray] = None) -> dict:
    """单图指标。``pred_mask`` 为 0/255 或 0/1 均可。"""
    gt = gt_mask > 0
    pr = pred_mask > 0
    inter = int((gt & pr).sum())
    dice = 2.0 * inter / max(int(gt.sum()) + int(pr.sum()), 1)
    iou = inter / max(int((gt | pr).sum()), 1)
    recall = inter / max(int(gt.sum()), 1)

    n_pred, labels = cv2.connectedComponents(pr.astype(np.uint8))
    if gt_instance is not None:
        # 与实验一致：预测连通域数 / GT 实例数
        gt_components = max(len(np.unique(gt_instance)) - 1, 1)
    else:
        # 没有实例 ID 时退回 GT 连通域数，避免整批 Fragmentation 变成 NaN
        gt_components = max(cv2.connectedComponents(gt.astype(np.uint8))[0] - 1, 1)
    frag = (n_pred - 1) / gt_components

    inst_rec = float("nan")
    if gt_instance is not None:
        inst_ids = [int(i) for i in np.unique(gt_instance) if i != 0]
        hit = 0
        for iid in inst_ids:
            inst = gt_instance == iid
            if n_pred > 1:
                best = max(
                    int((inst & (labels == lid)).sum()) / max(int(inst.sum()), 1)
                    for lid in range(1, n_pred)
                )
            else:
                best = 0.0
            if best >= 0.5:
                hit += 1
        inst_rec = hit / len(inst_ids) if inst_ids else float("nan")

    return {
        "dice": float(dice),
        "iou": float(iou),
        "recall": float(recall),
        "instance_recall": inst_rec,
        "fragmentation": float(frag),
        "skeleton_recall": skeleton_recall(gt, pr),
        "boundary_error": boundary_error(gt, pr),
    }


SUMMARY_KEYS = (
    "dice", "iou", "recall", "instance_recall",
    "skeleton_recall", "fragmentation", "boundary_error",
)


def summarize(results: list[dict]) -> dict:
    """对逐图指标取 nanmean（与实验脚本一致）。"""
    import math

    summary = {}
    for key in SUMMARY_KEYS:
        values = [r[key] for r in results
                  if key in r and not (isinstance(r[key], float) and math.isnan(r[key]))]
        summary[key] = float(np.mean(values)) if values else float("nan")
    return summary
