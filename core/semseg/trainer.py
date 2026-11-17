# -*- coding: utf-8 -*-
"""语义分割训练器（纯 PyTorch，与实验 E 的超参对齐）。

实验最优配置（``scratch_seg/configs/resunet34_512_e_best.py``）：
    AdamW lr=1e-4（编码器 ×0.1）、wd=1e-4、batch=1 + 梯度累积 4、
    LinearLR(1e-3 → 1, 500 iter) + PolyLR(power=0.9, eta_min=1e-6)、AMP、
    BCE(pos_weight=5)+Dice+clDice、512 patch 缺陷中心采样、8000 iter。

平台是 **epoch 制**：1 epoch = 遍历一遍 ``len(images) × patches_per_image``
个 patch（默认约 864），默认 12 个 epoch ≈ 1 万微批，与实验预算相当。

产物（与 YOLO 任务保持同名，方便现有清理/归档/继承逻辑复用）：
    <output_dir>/best.pt       平台格式 {'model':..., 'meta':...}
    <output_dir>/last.pt
    <output_dir>/results.csv   epoch,dice,iou,recall,train_loss,val_loss,lr
"""

import csv
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import load_weights, save_platform_checkpoint
from .dataset import build_datasets, load_manifest
from .infer import IMAGENET_MEAN, IMAGENET_STD, resolve_device
from .losses import (build_criterion, confusion_from_pred, label_long,
                     metrics_from_confusion)
from .model import build_model

_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)


def _normalize(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """0~255 的 (B,3,H,W) → ImageNet 归一化，并与实验一样做 BGR→RGB。"""
    x = batch.to(device, non_blocking=True).float()
    x = x[:, [2, 1, 0], :, :]
    return (x - _MEAN.to(device)) / _STD.to(device)


def _poly_lr(base_lr: float, eta_min: float, progress: float, power: float = 0.9) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return eta_min + (base_lr - eta_min) * (1.0 - progress) ** power


def _crop_logits_center(logits: torch.Tensor, valid: int) -> torch.Tensor:
    """把 stride-2 logits 中心裁到有效区（Halo 训练：loss/指标只看中心）。"""
    if valid <= 0:
        return logits
    h, w = logits.shape[-2:]
    vh = min(valid, h * 2)
    vw = min(valid, w * 2)
    th, tw = max(1, vh // 2), max(1, vw // 2)
    if th == h and tw == w:
        return logits
    y0, x0 = (h - th) // 2, (w - tw) // 2
    return logits[..., y0:y0 + th, x0:x0 + tw]


def _build_optimizer(model: torch.nn.Module, lr: float, encoder_lr_mult: float,
                     weight_decay: float) -> torch.optim.Optimizer:
    encoder, others = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone.enc"):
            encoder.append(param)
        else:
            others.append(param)
    groups = [
        {"params": others, "lr": lr},
        {"params": encoder, "lr": lr * encoder_lr_mult},
    ]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             threshold: float = 0.5, num_classes: int = 1,
             criterion: Optional[torch.nn.Module] = None,
             valid_size: int = 0) -> Dict[str, float]:
    """patch 级验证：全局累计 TP/FP/FN（比逐 batch 平均稳定）。

    多分类时把预测按阈值合成一张前景图后累计混淆矩阵，
    同时给出合并二分类指标（dice/iou/recall）与 mIoU/mDice/各类 Dice，
    这样监控页原有的 Dice/IoU/Recall 曲线在多分类下仍然可读。
    """
    model.eval()
    multiclass = int(num_classes) > 1
    if criterion is None:
        criterion = build_criterion("multiclass" if multiclass else "binary",
                                    num_classes).to(device)
    loss_sum = 0.0
    count = 0
    tp = fp = fn = 0
    conf = torch.zeros(int(num_classes), int(num_classes), dtype=torch.float64)

    for batch in loader:
        image = _normalize(batch["image"], device)
        mask = batch["mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(image)
            logits = _crop_logits_center(logits, valid_size)
            logits_full = F.interpolate(logits.float(), size=mask.shape[-2:],
                                        mode="bilinear", align_corners=False)
            loss = criterion(logits, mask.float() if not multiclass else mask)
        loss_sum += float(loss.item())
        count += 1

        if multiclass:
            label = label_long(logits_full, mask, ignore_index=255)
            prob = torch.softmax(logits_full, dim=1)
            fg_prob = 1.0 - prob[:, 0]
            pred_cls = prob.argmax(dim=1)
            pred_idx = torch.where(fg_prob > float(threshold), pred_cls,
                                   torch.zeros_like(pred_cls))
            conf += confusion_from_pred(pred_idx, label, num_classes).double().cpu()
        else:
            prob = torch.sigmoid(logits_full) > float(threshold)
            target = mask > 127
            tp += int((prob & target).sum().item())
            fp += int((prob & ~target).sum().item())
            fn += int((~prob & target).sum().item())

    if multiclass:
        metrics = metrics_from_confusion(conf)
        metrics["loss"] = loss_sum / max(count, 1)
        return metrics

    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "recall": float(recall),
        "loss": loss_sum / max(count, 1),
    }


def run_semseg_train(
    manifest_path: str,
    output_dir: str,
    epochs: int = 12,
    patch_size: int = 512,
    patches_per_image: int = 4,
    batch_size: int = 1,
    accum_steps: int = 4,
    lr: float = 1e-4,
    encoder_lr_mult: float = 0.1,
    weight_decay: float = 1e-4,
    pos_weight: float = 5.0,
    cldice_weight: float = 1.0,
    warmup_iters: int = 500,
    workers: int = 2,
    device: str = "auto",
    threshold: float = 0.5,
    # 多分类：num_classes = 背景 + N 个缺陷类（与 masks_ids 的像素值一致）
    num_classes: int = 1,
    fg_class_weight: float = 20.0,
    class_names: Optional[list] = None,
    # Halo：输入 = patch_size + 2×halo，loss/指标只看中心 patch_size
    halo: int = 0,
    patience: int = 0,
    # 增广（沿用平台字段）
    flipud: float = 0.5,
    fliplr: float = 0.5,
    color_aug_mode: str = "hsv",
    hsv_v: float = 0.4,
    rgb_gain_p: float = 0.0,
    rgb_gain_max: float = 0.0,
    degrees: float = 0.0,
    scale: float = 0.1,
    translate: float = 0.0,
    # 训练来源
    pretrained: bool = True,
    base_checkpoint: Optional[str] = None,
    seed: int = 0,
    # 回调
    epoch_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """执行语义分割训练，返回 ``{best_pt, last_pt, results_csv, train_dir, best_dice}``。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifest = load_manifest(manifest_path)
    patch_size = int(manifest.get("patch_size", patch_size) or patch_size)
    # manifest 里记录的类别数优先（数据准备时按项目类别算好）
    manifest_classes = int(manifest.get("num_classes", 1) or 1)
    num_classes = max(int(num_classes or 1), manifest_classes)
    multiclass = num_classes > 1
    if class_names is None:
        class_names = list(manifest.get("class_names") or [])
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    dev = resolve_device(device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    print(f"[semseg] 设备={dev} patch={patch_size} epochs={epochs} "
          f"模式={'多分类' if multiclass else '二分类'} num_classes={num_classes}")

    augment_kwargs = dict(
        flipud=flipud, fliplr=fliplr,
        color_aug_mode=color_aug_mode, hsv_v=hsv_v,
        rgb_gain_p=rgb_gain_p, rgb_gain_max=rgb_gain_max,
        degrees=degrees, scale=scale, translate=translate,
        halo=max(0, int(halo)),
        seed=seed,
    )
    train_ds = build_datasets(manifest_path, is_train=True,
                             patches_per_image=patches_per_image, **augment_kwargs)
    val_ds = build_datasets(manifest_path, is_train=False,
                           patches_per_image=1, **augment_kwargs)
    if len(val_ds) == 0:
        raise ValueError("验证集为空：请至少提供 2 张有缺陷的标注图")

    train_loader = DataLoader(train_ds, batch_size=max(1, int(batch_size)),
                              shuffle=True, num_workers=max(0, int(workers)),
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(1, int(batch_size)),
                            shuffle=False, num_workers=max(0, int(workers)),
                            drop_last=False)
    print(f"[semseg] train patches={len(train_ds)} val patches={len(val_ds)} "
          f"iters/epoch={len(train_loader)}")

    # Halo 相关尺寸（数据集里已把标签裁到有效区）
    valid_size = int(train_ds.valid_size)
    halo = int(train_ds.halo)
    halo_requested = int(getattr(train_ds, "halo_requested", halo))
    halo_note = str(getattr(train_ds, "halo_note", "") or "")
    input_size = int(train_ds.input_size)
    if halo > 0:
        print(f"[semseg] Halo={halo}：网络输入 {input_size}，有效区 {valid_size}（loss/指标只看中心）")
    elif halo_note:
        print(f"[semseg] {halo_note}")

    model = build_model(num_classes=num_classes, pretrained=pretrained)
    if base_checkpoint:
        # 分类数变了（二值 ⇄ 多分类、类别数变动）时只迁就形状一致的权重
        missing, unexpected = load_weights(
            model, base_checkpoint, strict=False, verbose=False, skip_mismatch=True)
        print(f"[semseg] 继承权重：{base_checkpoint} "
              f"（missing={len(missing)} unexpected={len(unexpected)}）")
    model.to(dev)

    optimizer = _build_optimizer(model, lr, encoder_lr_mult, weight_decay)
    criterion = build_criterion(
        "multiclass" if multiclass else "binary",
        num_classes=num_classes,
        pos_weight=pos_weight,
        cldice_weight=cldice_weight,
        fg_class_weight=fg_class_weight,
    ).to(dev)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    use_amp = dev.type == "cuda"

    iters_per_epoch = max(1, len(train_loader))
    total_iters = max(1, iters_per_epoch * int(epochs))
    warmup_iters = max(0, min(int(warmup_iters), total_iters // 5))

    results_path = output / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["epoch", "dice", "iou", "recall",
                         "miou", "mdice", "train_loss", "val_loss", "lr"])

    best_dice = -1.0
    best_path = output / "best.pt"
    last_path = output / "last.pt"
    best_epoch = -1
    global_iter = 0
    stale_epochs = 0
    started = time.time()
    cancelled = False

    for epoch in range(int(epochs)):
        model.train()
        train_ds.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        micro_in_accum = 0

        for step, batch in enumerate(train_loader):
            if cancel_check is not None and global_iter % 20 == 0:
                try:
                    if cancel_check():
                        print("[semseg] 收到取消信号，提前结束训练")
                        cancelled = True
                        break
                except Exception as exc:      # 取消检查失败不阻塞训练
                    print(f"[semseg] cancel_check 异常：{exc}")

            image = _normalize(batch["image"], dev)
            mask = batch["mask"].to(dev)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp):
                logits = model(image)
                # Halo：只对中心有效区算 loss（logits 是 stride-2，按有效区裁）
                logits = _crop_logits_center(logits, valid_size)
                loss = criterion(logits, mask.float() if not multiclass else mask)
                loss_scaled = loss / max(1, int(accum_steps))

            scaler.scale(loss_scaled).backward()
            micro_in_accum += 1
            running_loss += float(loss.item())
            global_iter += 1

            # 学习率（微批维度，与实验一致）
            if global_iter <= warmup_iters:
                factor = max(1e-3, global_iter / max(1, warmup_iters))
                for group, base in zip(optimizer.param_groups, (lr, lr * encoder_lr_mult)):
                    group["lr"] = base * factor
            else:
                progress = (global_iter - warmup_iters) / max(1, total_iters - warmup_iters)
                for group, base in zip(optimizer.param_groups, (lr, lr * encoder_lr_mult)):
                    group["lr"] = _poly_lr(base, 1e-6, progress)

            if micro_in_accum >= int(accum_steps):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                micro_in_accum = 0

            if step % 50 == 0:
                print(f"[semseg] epoch {epoch + 1}/{epochs} "
                      f"iter {step}/{iters_per_epoch} loss={float(loss.item()):.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if micro_in_accum > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if cancelled:
            break

        metrics = validate(model, val_loader, dev, threshold=threshold,
                           num_classes=num_classes, criterion=criterion,
                           valid_size=valid_size)
        avg_train_loss = running_loss / max(1, (step + 1))
        current_lr = optimizer.param_groups[0]["lr"]
        miou = float(metrics.get("miou") or 0.0)
        mdice = float(metrics.get("mdice") or 0.0)

        with results_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                epoch + 1, f"{metrics['dice']:.6f}", f"{metrics['iou']:.6f}",
                f"{metrics['recall']:.6f}", f"{miou:.6f}", f"{mdice:.6f}",
                f"{avg_train_loss:.6f}",
                f"{metrics['loss']:.6f}", f"{current_lr:.8f}",
            ])

        improved = metrics["dice"] > best_dice
        if improved:
            best_dice = metrics["dice"]
            best_epoch = epoch + 1
            stale_epochs = 0
            save_platform_checkpoint(model, best_path, meta={
                "epoch": epoch + 1,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "recall": metrics["recall"],
                "miou": miou,
                "mdice": mdice,
                "class_dice": metrics.get("class_dice") or [],
                "patch_size": patch_size,
                "threshold": threshold,
                "halo": halo,
                "halo_requested": halo_requested,
                "halo_note": halo_note,
                "input_size": input_size,
                "num_classes": num_classes,
                "mode": "multiclass" if multiclass else "binary",
                "class_names": class_names,
                "arch": "resunet34_cldice",
            })
        else:
            stale_epochs += 1

        save_platform_checkpoint(model, last_path, meta={
            "epoch": epoch + 1,
            "dice": metrics["dice"],
            "miou": miou,
            "mdice": mdice,
            "patch_size": patch_size,
            "halo": halo,
            "halo_requested": halo_requested,
            "halo_note": halo_note,
            "input_size": input_size,
            "num_classes": num_classes,
            "mode": "multiclass" if multiclass else "binary",
            "class_names": class_names,
            "arch": "resunet34_cldice",
        })

        print(f"[semseg] epoch {epoch + 1}/{epochs} "
              f"dice={metrics['dice']:.4f} iou={metrics['iou']:.4f} "
              f"recall={metrics['recall']:.4f} val_loss={metrics['loss']:.4f}"
              + (f" miou={miou:.4f} mdice={mdice:.4f}" if multiclass else "")
              + (" *best*" if improved else ""))

        if epoch_callback is not None:
            try:
                epoch_callback({
                    "epoch": epoch,
                    "total_epochs": int(epochs),
                    "train_seg_loss": float(avg_train_loss),
                    "val_seg_loss": float(metrics["loss"]),
                    "val_dice": float(metrics["dice"]),
                    "val_iou": float(metrics["iou"]),
                    "val_recall": float(metrics["recall"]),
                    "val_miou": miou,
                    "val_mdice": mdice,
                    "best_fitness": float(best_dice),
                    "mAP50_B": float(metrics["dice"]),
                    "lr": float(current_lr),
                })
            except Exception as exc:
                print(f"[semseg] epoch_callback 异常：{exc}")

        if progress_callback:
            try:
                progress_callback(epoch + 1, int(epochs), "训练中")
            except Exception:
                pass

        if patience and stale_epochs >= int(patience):
            print(f"[semseg] 连续 {patience} 个 epoch 无提升，提前停止")
            break

    elapsed = time.time() - started
    print(f"[semseg] 训练结束，用时 {elapsed / 60:.1f} min，best dice={best_dice:.4f}"
          f" @ epoch {best_epoch}")

    return {
        "best_pt": str(best_path) if best_path.exists() else None,
        "last_pt": str(last_path) if last_path.exists() else None,
        "results_csv": str(results_path),
        "train_dir": str(output),
        "best_dice": float(best_dice),
        "best_epoch": int(best_epoch),
        "cancelled": bool(cancelled),
    }
