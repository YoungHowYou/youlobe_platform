# -*- coding: utf-8 -*-
"""可选的实例分割掩膜损失（只替换掩膜项，box/cls/DFL 与标签分配保持原版）。

设计要点：
- ``default`` 不注入任何 criterion，完全走 Ultralytics 原版 ``v8SegmentationLoss``。
- 自定义项继承 ``v8SegmentationLoss``，只覆写 ``single_mask_loss``；检测框、
  分类、DFL、任务对齐标签分配全部沿用父类实现。
- ``focal`` / ``boundary`` 是逐像素损失的替换，沿用原版「按框面积归一」口径；
  ``bce_dice`` 按框内像素均值归一，让 BCE 与 Dice 两项量级可比；
  ``tversky`` / ``focal_tversky`` 是区域型损失，按实例累加、天然与尺度无关。
- 自定义项不再叠加小实例加权，衡量小目标请用 ``tools/eval_size_buckets.py``
  的尺寸分桶指标。
"""

from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.ops import crop_mask


SEGMENTATION_LOSS_OPTIONS: Tuple[str, ...] = (
    "default",
    "bce_dice",
    "focal",
    "tversky",
    "focal_tversky",
    "boundary",
)

DEFAULT_SEG_LOSS_PARAMS: Dict[str, float] = {
    "dice_weight": 0.5,          # bce_dice：Dice 占比（BCE 占 1 - dice_weight）
    "focal_gamma": 2.0,          # focal：难例聚焦强度
    "focal_alpha": 0.25,         # focal：正样本权重（缺陷像素稀缺时可调到 0.75）
    "tversky_alpha": 0.3,        # tversky：误检 FP 惩罚权重
    "tversky_beta": 0.7,         # tversky：漏检 FN 惩罚权重
    "focal_tversky_gamma": 0.75,  # focal_tversky：难例放大指数
    "boundary_weight": 2.0,      # boundary：边界像素额外权重（1.0 = 加倍）
}

_BOUNDARY_KERNEL = 3


def normalize_seg_loss_params(raw: Any) -> Dict[str, float]:
    """把前端/DB 传来的任意 dict 归一化成带边界截断的完整参数字典。"""
    params = dict(DEFAULT_SEG_LOSS_PARAMS)
    if isinstance(raw, dict):
        for key in params:
            value = raw.get(key)
            if value is None:
                continue
            try:
                params[key] = float(value)
            except (TypeError, ValueError):
                continue

    def _clip(name: str, low: float, high: float) -> None:
        params[name] = min(high, max(low, params[name]))

    _clip("dice_weight", 0.0, 1.0)
    _clip("focal_gamma", 0.0, 10.0)
    _clip("focal_alpha", 0.0, 1.0)
    _clip("tversky_alpha", 0.0, 1.0)
    _clip("tversky_beta", 0.0, 1.0)
    _clip("focal_tversky_gamma", 0.0, 5.0)
    _clip("boundary_weight", 0.0, 20.0)
    return params


class SelectableSegmentationLoss(v8SegmentationLoss):
    """保留 Ultralytics 检测损失，只替换实例掩膜项。"""

    def __init__(
        self,
        model,
        loss_type: str = "bce_dice",
        loss_params: Optional[Dict[str, float]] = None,
        tal_topk: int = 10,
        tal_topk2: Optional[int] = None,
    ):
        super().__init__(model, tal_topk, tal_topk2)
        self.loss_type = loss_type
        self.loss_params = normalize_seg_loss_params(loss_params)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        loss_type = self.loss_type
        params = self.loss_params

        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        target = gt_mask.float()
        valid = crop_mask(torch.ones_like(target), xyxy).float()
        pixel_count = valid.sum(dim=(1, 2)).clamp_min(1.0)
        probability = pred_mask.sigmoid()
        bce = F.binary_cross_entropy_with_logits(pred_mask, target, reduction="none")

        if loss_type == "bce_dice":
            bce_loss = (bce * valid).sum(dim=(1, 2)) / pixel_count
            intersection = (probability * target * valid).sum(dim=(1, 2))
            denominator = ((probability + target) * valid).sum(dim=(1, 2))
            dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
            dice_weight = params["dice_weight"]
            return ((1.0 - dice_weight) * bce_loss + dice_weight * dice_loss).sum()

        if loss_type == "focal":
            gamma = params["focal_gamma"]
            alpha = params["focal_alpha"]
            p_t = probability * target + (1.0 - probability) * (1.0 - target)
            alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
            focal = alpha_t * (1.0 - p_t).pow(gamma) * bce
            # 与父类同口径：按框面积归一，便于和默认项直接对比
            return (crop_mask(focal, xyxy).mean(dim=(1, 2)) / area).sum()

        if loss_type in {"tversky", "focal_tversky"}:
            alpha_fp = params["tversky_alpha"]
            beta_fn = params["tversky_beta"]
            pred_valid = probability * valid
            target_valid = target * valid
            true_positive = (pred_valid * target_valid).sum(dim=(1, 2))
            false_positive = (pred_valid * (1.0 - target_valid)).sum(dim=(1, 2))
            false_negative = ((1.0 - pred_valid) * target_valid).sum(dim=(1, 2))
            score = (true_positive + 1.0) / (
                true_positive + alpha_fp * false_positive + beta_fn * false_negative + 1.0
            )
            loss = 1.0 - score
            if loss_type == "focal_tversky":
                loss = loss.pow(params["focal_tversky_gamma"])
            return loss.sum()

        if loss_type == "boundary":
            extra_weight = params["boundary_weight"]
            target_map = (target * valid).unsqueeze(1)
            dilated = F.max_pool2d(target_map, kernel_size=_BOUNDARY_KERNEL, stride=1, padding=1)
            eroded = 1.0 - F.max_pool2d(
                1.0 - target_map, kernel_size=_BOUNDARY_KERNEL, stride=1, padding=1
            )
            boundary = (dilated - eroded).squeeze(1).clamp_(0.0, 1.0) * valid
            weighted_bce = bce * (valid + extra_weight * boundary)
            # 与父类同口径：按框面积归一
            return (crop_mask(weighted_bce, xyxy).mean(dim=(1, 2)) / area).sum()

        raise ValueError(f"不支持的实例分割损失选项: {loss_type}")


def build_segmentation_criterion(
    model,
    loss_type: str,
    loss_params: Optional[Dict[str, float]] = None,
):
    """构造 criterion；YOLO26 的 end2end 头需要用 ``E2ELoss`` 包装。"""
    if loss_type not in SEGMENTATION_LOSS_OPTIONS or loss_type == "default":
        raise ValueError(f"不支持的实例分割损失选项: {loss_type}")

    factory = partial(
        SelectableSegmentationLoss,
        loss_type=loss_type,
        loss_params=normalize_seg_loss_params(loss_params),
    )
    head = model.model[-1]
    if getattr(head, "one2one_cv2", None) is not None:
        return E2ELoss(model, loss_fn=factory)
    return factory(model)