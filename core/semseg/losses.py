# -*- coding: utf-8 -*-
"""语义分割 Loss（与 scratch_seg/scratchseg/losses.py 数值口径一致）。

实验结论：最优组合是 ``BCE(pos_weight=5) + Dice + clDice(weight=1.0)``。
注意 **pos_weight 是主导变量**（20 → 5 才是 Dice 从 0.558 涨到 0.699 的原因），
clDice 只能做辅助项，权重放大到 3 倍会概率双峰化。
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to01(seg_logit: torch.Tensor, seg_label: torch.Tensor) -> torch.Tensor:
    """标签归一化到 {0,1} 的 (B,1,H,W)，并对齐到 logits 分辨率（stride-2）。"""
    seg_label = seg_label.to(device=seg_logit.device, dtype=seg_logit.dtype)
    if seg_label.dim() == 3:
        seg_label = seg_label.unsqueeze(1)
    seg_label = (seg_label > 127).float()
    if seg_label.shape[-2:] != seg_logit.shape[-2:]:
        seg_label = F.interpolate(
            seg_label, size=seg_logit.shape[-2:], mode="nearest"
        )
    return seg_label


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-img, 3, 1, 1)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, 3, 1, 1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeleton(img: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """clDice 论文的可微软骨架（迭代 soft-erode + 残差）。"""
    skel = torch.relu(img - soft_open(img))
    for _ in range(iters):
        img = soft_erode(img)
        delta = torch.relu(img - soft_open(img))
        skel = skel + torch.relu(delta - skel * delta)
    return skel


class BCEDiceLoss(nn.Module):
    """基线 A：BCE(pos_weight) + soft-Dice。"""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 pos_weight: float = 20.0, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, seg_logit: torch.Tensor,
                seg_label: torch.Tensor, **kwargs) -> torch.Tensor:
        target = _to01(seg_logit, seg_label)
        bce = F.binary_cross_entropy_with_logits(
            seg_logit, target,
            pos_weight=torch.tensor(float(self.pos_weight),
                                    device=seg_logit.device),
        )
        prob = torch.sigmoid(seg_logit)
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - (2 * inter + self.smooth) / (denom + self.smooth)
        return self.bce_weight * bce + self.dice_weight * dice.mean()


class BCEDiceClDiceLoss(nn.Module):
    """实验 E：BCE(pos_weight=5) + Dice + clDice。

    Args:
        pos_weight: BCE 正样本权重。**默认 5**，与最优实验一致。
        cldice_weight: clDice 项权重，默认 1.0；不要放大（3 倍会双峰化）。
        skel_iters: soft-skeleton 迭代次数，实验用 8。
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 cldice_weight: float = 1.0, pos_weight: float = 5.0,
                 smooth: float = 1.0, skel_iters: int = 8,
                 class_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.pos_weight = pos_weight
        self.smooth = smooth
        self.skel_iters = skel_iters
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.float())
        else:
            self.class_weight = None

    def forward(self, seg_logit: torch.Tensor,
                seg_label: torch.Tensor, **kwargs) -> torch.Tensor:
        target = _to01(seg_logit, seg_label)
        bce = F.binary_cross_entropy_with_logits(
            seg_logit, target,
            pos_weight=torch.tensor(float(self.pos_weight),
                                    device=seg_logit.device),
        )
        prob = torch.sigmoid(seg_logit)
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - (2 * inter + self.smooth) / (denom + self.smooth)

        skel_p = soft_skeleton(prob, self.skel_iters)
        skel_g = soft_skeleton(target, self.skel_iters)
        t_prec = (skel_p * target).sum() / (skel_p.sum() + 1e-8)
        t_rec = (prob * skel_g).sum() / (skel_g.sum() + 1e-8)
        cldice = 1.0 - 2 * t_prec * t_rec / (t_prec + t_rec + 1e-8)

        return (self.bce_weight * bce
                + self.dice_weight * dice.mean()
                + self.cldice_weight * cldice)


def dice_iou_recall_from_logits(seg_logit: torch.Tensor, seg_label: torch.Tensor,
                               threshold: float = 0.5) -> dict:
    """训练期验证用的批量像素指标（patch 级，不含滑窗整图统计）。"""
    target = _to01(seg_logit, seg_label) > 0.5
    pred = torch.sigmoid(seg_logit) > threshold
    inter = (pred & target).sum().item()
    pred_sum = pred.sum().item()
    gt_sum = target.sum().item()
    dice = 2 * inter / max(pred_sum + gt_sum, 1)
    iou = inter / max(pred_sum + gt_sum - inter, 1)
    recall = inter / max(gt_sum, 1)
    return {"dice": dice, "iou": iou, "recall": recall}


# ---------------------------------------------------------------------------
# 多分类（背景 + N 个缺陷类）
# ---------------------------------------------------------------------------
def label_long(seg_logit: torch.Tensor, seg_label: torch.Tensor,
               ignore_index: int = 255) -> torch.Tensor:
    """标签归一化成 (B,H,W) 的 long 类别图，并对齐到 logits 分辨率。

    多分类的标签是**类别索引**（0=背景，1..N=缺陷类，与 ``masks_ids`` 一致），
    不是 0/255 的灰度 mask，因此这里不能沿用 ``_to01``。
    """
    label = seg_label
    if label.dim() == 4:                     # (B,1,H,W)
        label = label[:, 0]
    label = label.to(device=seg_logit.device)
    if label.dtype.is_floating_point:
        label = label.round()
    label = label.long()
    if label.shape[-2:] != seg_logit.shape[-2:]:
        label = F.interpolate(label[:, None].float(), size=seg_logit.shape[-2:],
                              mode="nearest")[:, 0].long()
    # 只保留合法类别，其余（含 255 填充）视为 ignore
    num_classes = int(seg_logit.shape[1])
    label = torch.where((label >= 0) & (label < num_classes), label,
                        torch.full_like(label, int(ignore_index)))
    return label


class CEDiceLoss(nn.Module):
    """多分类语义分割 Loss：CE(class_weight) + 前景 soft-Dice。

    口径来源：``scratch_seg/configs/resunet34_512_mc3.py``（对照实验 B 组）
      - ``num_classes = 背景 + N 个缺陷类``
      - ``class_weight = [1, 20, 20, ...]``：缺陷占比 ~0.004%，
        背景必须压到 1，否则模型直接学成“全背景”
      - Dice 只在**前景类**上逐类计算后取平均（背景 Dice 恒接近 1，加进来会稀释）

    对照实验结论（test）：合并二分类 Recall 0.772 / Dice 0.700，
    类间混淆 < 10%，因此多分类不会牺牲检出能力。
    """

    def __init__(self, num_classes: int = 2, fg_class_weight: float = 20.0,
                 class_weight: Optional[Sequence[float]] = None,
                 ce_weight: float = 1.0, dice_weight: float = 1.0,
                 ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.fg_class_weight = float(fg_class_weight)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ignore_index = int(ignore_index)
        self.smooth = float(smooth)
        if class_weight is not None:
            weight = torch.as_tensor(list(class_weight), dtype=torch.float32)
        else:
            weight = torch.ones(self.num_classes, dtype=torch.float32)
            weight[1:] = float(fg_class_weight)
        self.register_buffer("class_weight", weight)

    def forward(self, seg_logit: torch.Tensor, seg_label: torch.Tensor,
                **kwargs) -> torch.Tensor:
        label = label_long(seg_logit, seg_label, self.ignore_index)
        ce = F.cross_entropy(seg_logit, label,
                             weight=self.class_weight.to(seg_logit.device),
                             ignore_index=self.ignore_index)

        prob = torch.softmax(seg_logit, dim=1)
        valid = (label != self.ignore_index).float()
        label_safe = label.clamp(min=0)
        dice_terms = []
        for c in range(1, self.num_classes):
            p = prob[:, c] * valid
            t = ((label_safe == c) & (valid > 0)).float()
            inter = (p * t).sum(dim=(1, 2))
            denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice_terms.append(1.0 - (2 * inter + self.smooth) / (denom + self.smooth))
        dice = torch.stack(dice_terms).mean() if dice_terms else ce.new_zeros(())
        return self.ce_weight * ce + self.dice_weight * dice


def build_criterion(mode: str = "binary", num_classes: int = 1,
                    pos_weight: float = 5.0, cldice_weight: float = 1.0,
                    fg_class_weight: float = 20.0) -> nn.Module:
    """按分割模式构造 Loss（binary = 二值缺陷，multiclass = 背景+多类）。"""
    if int(num_classes) > 1 or str(mode).lower() == "multiclass":
        return CEDiceLoss(num_classes=max(2, int(num_classes)),
                          fg_class_weight=fg_class_weight)
    return BCEDiceClDiceLoss(pos_weight=pos_weight, cldice_weight=cldice_weight)


def confusion_from_pred(pred_idx: torch.Tensor, label: torch.Tensor,
                        num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """把一批预测类别图 (B,H,W) 与标签累加进 num_classes 混淆矩阵。

    行 = GT，列 = 预测；``ignore_index`` 像素不参与统计。
    """
    if label.dim() == 4:
        label = label[:, 0]
    label = label.to(pred_idx.device).long()
    if label.shape[-2:] != pred_idx.shape[-2:]:
        label = F.interpolate(label[:, None].float(), size=pred_idx.shape[-2:],
                              mode="nearest")[:, 0].long()
    valid = label != ignore_index
    flat = label[valid].clamp(min=0) * num_classes + pred_idx[valid].clamp(min=0)
    return torch.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes)


def confusion_from_logits(pred: torch.Tensor, label: torch.Tensor,
                          num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """把一批 (B,C,H,W) logits 与 (B,H,W) 标签累加进 num_classes 混淆矩阵。"""
    return confusion_from_pred(pred.argmax(dim=1), label, num_classes, ignore_index)


def metrics_from_confusion(conf: torch.Tensor) -> dict:
    """由混淆矩阵算合并二分类指标 + 多分类 mIoU/mDice + 各类 Dice。

    「合并二分类」口径与对照实验一致：任一缺陷类判为前景即算 TP
    （等价于把这些行的预测列求和后与所有缺陷列比较）。
    """
    conf = conf.double()
    num_classes = int(conf.shape[0])
    fg_rows = conf[1:, :].sum(dim=0)           # GT=缺陷，按预测列聚合
    tp = float(fg_rows[1:].sum().item())
    fn = float(fg_rows[0].item())
    fp = float(conf[0, 1:].sum().item())
    tn = float(conf[0, 0].item())
    dice = 2 * tp / max(2 * tp + fp + fn, 1e-9)
    iou = tp / max(tp + fp + fn, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    precision = tp / max(tp + fp, 1e-9)

    per_class_dice, per_class_iou = [], []
    for c in range(1, num_classes):
        ctp = float(conf[c, c].item())
        cfp = float(conf[:, c].sum().item()) - ctp
        cfn = float(conf[c, :].sum().item()) - ctp
        per_class_dice.append(2 * ctp / max(2 * ctp + cfp + cfn, 1e-9))
        per_class_iou.append(ctp / max(ctp + cfp + cfn, 1e-9))
    miou = sum(per_class_iou) / len(per_class_iou) if per_class_iou else 0.0
    mdice = sum(per_class_dice) / len(per_class_dice) if per_class_dice else 0.0
    return {
        "dice": dice, "iou": iou, "recall": recall, "precision": precision,
        "tn": tn, "miou": miou, "mdice": mdice,
        "class_dice": per_class_dice, "class_iou": per_class_iou,
    }
