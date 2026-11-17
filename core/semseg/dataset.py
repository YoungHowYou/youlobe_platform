# -*- coding: utf-8 -*-
"""语义分割数据集：缺陷中心 patch 采样。

实验结论：缺陷面积中位数约 600 / 16.7M 像素（0.0036%），**随机/全覆盖裁剪
几乎采不到正样本**，因此训练 patch 必须以缺陷像素为中心随机裁剪。

这里与实验 ``DefectPatchDataset`` 保持一致：

- 训练：随机选一个缺陷像素 → 以它为中心 ±patch/4 随机偏移 → 裁 ``patch_size``
  → 随机水平/垂直翻转（p=0.5）→ 颜色增广
- 验证：取缺陷**质心**的固定 patch（可重复，便于曲线对比）
- 小于 patch 的图用 0 在右下补齐（与推理端 pad 口径一致）
- 标签是 0/255 的灰度 mask（``_to01`` 按 >127 判定，务必不要存成 0/1）

数据集划分在 ``prepare_semantic_dataset`` 里**按图**完成，避免同一张原图的
patch 同时出现在 train 和 val。
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .infer import read_image_gray


def _resize_keep(img: np.ndarray, mask: np.ndarray, scale: float,
                 rng: np.random.Generator):
    """按比例缩放图像与 mask（最近邻给 mask）。"""
    if abs(scale - 1.0) < 1e-3:
        return img, mask
    h, w = img.shape[:2]
    nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask2 = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return img2, mask2


def _apply_brightness(img_gray: np.ndarray, gain: float) -> np.ndarray:
    """单通道亮度乘性增益（灰度图的 HSV-V / RGB 逐通道增益都退化成它）。"""
    return np.clip(img_gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def probe_image_sizes(items: list[dict], limit: int = 2000) -> tuple[int, int]:
    """探测样本图像尺寸，返回 ``(最大边, 最小边)``；全都读不到时返回 ``(0, 0)``。

    manifest 里带 ``width`` / ``height`` 时直接复用；老 manifest 没有这两个字段，
    就只读图像文件头（PIL），实在拿不到才解码整图，避免为一次判断把整批图读进内存。
    """
    max_side = min_side = 0
    for item in items[:limit]:
        w = int(item.get("width") or 0)
        h = int(item.get("height") or 0)
        if w <= 0 or h <= 0:
            path = item.get("image")
            if not path:
                continue
            try:
                from PIL import Image
                with Image.open(path) as im:
                    w, h = im.size
            except Exception:
                try:
                    img = read_image_gray(path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                except Exception:
                    continue
        side_max, side_min = max(int(w), int(h)), min(int(w), int(h))
        max_side = max(max_side, side_max)
        min_side = side_min if min_side == 0 else min(min_side, side_min)
    return max_side, min_side


def resolve_halo(items: list[dict], valid_size: int, halo: int) -> tuple[int, str]:
    """按图像尺寸决定 halo 是否真的生效。

    ``图像最大边 <= 有效区`` 时整图就是一个 patch、不切块，halo 取不到任何真实邻居
    上下文（只会补出黑边，还会把图像挤到角落），因此自动关闭。

    Returns:
        ``(生效的 halo, 说明文字)``；说明文字为空表示没有改动。
    """
    halo = max(0, int(halo))
    halo -= halo % 2
    if halo <= 0:
        return 0, ""
    max_side, _ = probe_image_sizes(items)
    if max_side and int(valid_size) >= max_side:
        return 0, (f"图像最大边 {max_side} ≤ 有效区 {valid_size}"
                   f"（整图即一个 patch），已自动关闭 Halo")
    return halo, ""


def resolve_halo_for_manifest(manifest_path: str | Path, valid_size: int,
                              halo: int) -> tuple[int, str]:
    """``resolve_halo`` 的 manifest 版本（训练任务落库用，口径与数据集一致）。"""
    manifest = load_manifest(manifest_path)
    items = list(manifest.get("train") or []) + list(manifest.get("val") or [])
    return resolve_halo(items, valid_size, halo)


class DefectPatchDataset(Dataset):
    """缺陷中心 patch 数据集。

    Args:
        items: manifest 中的样本列表，每项至少包含 ``image`` / ``mask`` / ``points``
        patch_size: 裁剪尺寸
        is_train: 训练模式（随机中心 + 增广）；验证模式取质心
        patches_per_image: 每图重复次数（训练时扩大 epoch 长度）
        flipud / fliplr: 翻转概率
        color_aug_mode: ``hsv``（仅亮度 v 生效）/ ``rgb``（逐通道增益）/ ``off``
        hsv_v / rgb_gain_p / rgb_gain_max_*: 颜色增广强度
        degrees: 旋转角度上限（默认 0，实验未使用）
        scale: 随机缩放上限（相对值，默认 0.1）
        translate: 随机平移比例（默认 0，实验用 ±patch/4 的中心抖动代替）
    """

    def __init__(
        self,
        items: list[dict],
        patch_size: int = 512,
        is_train: bool = True,
        patches_per_image: int = 4,
        flipud: float = 0.5,
        fliplr: float = 0.5,
        color_aug_mode: str = "hsv",
        hsv_v: float = 0.4,
        rgb_gain_p: float = 0.0,
        rgb_gain_max: float = 0.0,
        degrees: float = 0.0,
        scale: float = 0.1,
        translate: float = 0.0,
        num_classes: int = 1,
        halo: int = 0,
        seed: int = 0,
    ):
        if not items:
            raise ValueError("语义分割数据集为空")
        self.items = items
        self.patch_size = int(patch_size)        # 中心有效区边长（Loss/指标只看这块）
        self.num_classes = int(num_classes)
        self.multiclass = self.num_classes > 1
        # Halo：四周各扩 halo 像素作为上下文送进网络；必须偶数（logits 是 stride-2）
        self.valid_size = self.patch_size
        self.halo_requested = max(0, int(halo))
        self.halo_requested -= self.halo_requested % 2
        # 整图塞得进有效区 → 没有邻居上下文可取，halo 只补黑边，自动关闭
        self.halo, self.halo_note = resolve_halo(items, self.valid_size, self.halo_requested)
        self.input_size = self.valid_size + 2 * self.halo
        self.is_train = bool(is_train)
        self.patches_per_image = max(1, int(patches_per_image)) if is_train else 1
        self.flipud = float(flipud)
        self.fliplr = float(fliplr)
        self.color_aug_mode = str(color_aug_mode or "hsv").lower()
        self.hsv_v = float(hsv_v)
        self.rgb_gain_p = float(rgb_gain_p)
        self.rgb_gain_max = float(rgb_gain_max)
        self.degrees = float(degrees)
        self.scale = float(scale)
        self.translate = float(translate)
        self.seed = int(seed)
        self._epoch = 0

    # 让 DataLoader worker 每个 epoch 有不同的随机序列
    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items) * self.patches_per_image

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng((self.seed + self._epoch) * 1_000_003 + index)

    def _center(self, item: dict, rng: np.random.Generator):
        points_path = item.get("points")
        if points_path and Path(points_path).exists():
            pts = np.load(points_path)
        else:
            mask = read_image_gray(item["mask"])
            ys, xs = np.where(mask > 0)
            pts = np.stack([ys, xs], axis=1) if len(xs) else None

        if pts is None or len(pts) == 0:
            return None

        if self.is_train:
            cy, cx = pts[rng.integers(len(pts))]
            half = self.input_size // 2
            quarter = max(1, half // 2)
            cy = int(cy) + int(rng.integers(-quarter, quarter + 1))
            cx = int(cx) + int(rng.integers(-quarter, quarter + 1))
        else:
            cy = int(pts[:, 0].mean())
            cx = int(pts[:, 1].mean())
        return int(cy), int(cx)

    def _crop_pad(self, img: np.ndarray, mask: np.ndarray, cy: int, cx: int,
                  rng: Optional[np.random.Generator] = None):
        # 裁剪尺寸是 input_size（= 有效区 + 2×halo），小图右下补 0
        p = self.input_size
        h, w = img.shape[:2]
        half = p // 2
        if self.is_train and self.translate > 0 and rng is not None:
            cy = int(cy + self.translate * p * (rng.random() * 2 - 1))
            cx = int(cx + self.translate * p * (rng.random() * 2 - 1))

        if h >= p:
            y0 = int(np.clip(cy - half, 0, h - p))
        else:
            y0 = 0
        if w >= p:
            x0 = int(np.clip(cx - half, 0, w - p))
        else:
            x0 = 0

        crop = img[y0:y0 + p, x0:x0 + p]
        crop_mask = mask[y0:y0 + p, x0:x0 + p]
        if crop.shape[0] != p or crop.shape[1] != p:
            padded = np.zeros((p, p), dtype=img.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded
            padded_mask = np.zeros((p, p), dtype=mask.dtype)
            padded_mask[:crop_mask.shape[0], :crop_mask.shape[1]] = crop_mask
            crop_mask = padded_mask
        return np.ascontiguousarray(crop), np.ascontiguousarray(crop_mask)

    def __getitem__(self, index: int) -> dict:
        item_index = index // self.patches_per_image if self.is_train else index
        item = self.items[item_index]
        rng = self._rng(index)

        img = read_image_gray(item["image"])
        # 多分类读 masks_ids（像素值 = class_index + 1，0 = 背景）；
        # 二值读 masks_binary（0/255）
        mask_key = "mask_ids" if (self.multiclass and item.get("mask_ids")) else "mask"
        mask = read_image_gray(item[mask_key])
        if img is None or mask is None:
            raise RuntimeError(f"图像或 mask 读取失败：{item}")

        if self.is_train and self.scale > 0 and rng.random() < 0.5:
            factor = 1.0 + rng.uniform(-self.scale, self.scale)
            img, mask = _resize_keep(img, mask, factor, rng)

        center = self._center(item, rng)
        if center is None:
            # 没有缺陷点（OK 图）→ 随机位置裁剪，mask 全 0
            h, w = img.shape[:2]
            cy = int(rng.integers(0, max(1, h)))
            cx = int(rng.integers(0, max(1, w)))
        else:
            cy, cx = center

        patch, patch_mask = self._crop_pad(img, mask, cy, cx, rng)

        if self.is_train:
            if self.degrees > 0 and rng.random() < 0.5:
                angle = rng.uniform(-self.degrees, self.degrees)
                p = self.input_size
                mat = cv2.getRotationMatrix2D((p / 2, p / 2), angle, 1.0)
                patch = cv2.warpAffine(patch, mat, (p, p),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                patch_mask = cv2.warpAffine(patch_mask, mat, (p, p),
                                            flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if rng.random() < self.fliplr:
                patch = patch[:, ::-1]
                patch_mask = patch_mask[:, ::-1]
            if rng.random() < self.flipud:
                patch = patch[::-1]
                patch_mask = patch_mask[::-1]
            if self.color_aug_mode == "hsv" and self.hsv_v > 0:
                gain = 1.0 + rng.uniform(-self.hsv_v, self.hsv_v)
                patch = _apply_brightness(patch, gain)
            elif (self.color_aug_mode == "rgb" and self.rgb_gain_p > 0
                  and self.rgb_gain_max > 0 and rng.random() < self.rgb_gain_p):
                # 灰度输入下逐通道增益退化为亮度增益，这里仍保留独立采样语义
                gain = 1.0 + rng.uniform(-self.rgb_gain_max, self.rgb_gain_max)
                patch = _apply_brightness(patch, gain)
            patch = np.ascontiguousarray(patch)
            patch_mask = np.ascontiguousarray(patch_mask)

        # Halo：网络吃 input_size（含上下文），标签/指标只取中心有效区
        if self.halo > 0:
            off = (self.input_size - self.valid_size) // 2
            patch_mask = patch_mask[off:off + self.valid_size,
                                    off:off + self.valid_size]
            patch_mask = np.ascontiguousarray(patch_mask)

        # 灰度 → 3 通道；
        #   二值：mask 保持 0/255（_to01 依赖 >127）
        #   多分类：mask 是类别索引 0..N（CE 直接吃 long）
        image3 = np.repeat(patch[:, :, None], 3, axis=2).astype(np.float32)
        if self.multiclass:
            label_map = patch_mask.astype(np.int64)
            return {
                "image": torch.from_numpy(image3.transpose(2, 0, 1)),   # (3,H,W) 0~255
                "mask": torch.from_numpy(label_map),                     # (H,W) 类别索引
                "stem": item.get("stem", ""),
            }
        label = patch_mask.astype(np.float32)
        return {
            "image": torch.from_numpy(image3.transpose(2, 0, 1)),        # (3,H,W) 0~255
            "mask": torch.from_numpy(label[None]),                        # (1,H,W) 0/255
            "stem": item.get("stem", ""),
        }


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_datasets(manifest_path: str | Path, is_train: bool = True,
                   patches_per_image: int = 4, **augment) -> DefectPatchDataset:
    manifest = load_manifest(manifest_path)
    split = "train" if is_train else "val"
    items = manifest.get(split) or []
    if is_train and not items:
        items = manifest.get("train") or []
    # 显式传入的 num_classes 优先（二值评估时可强制 1），否则用 manifest 里的
    override = int(augment.pop("num_classes", 0) or 0)
    num_classes = override or int(manifest.get("num_classes", 1) or 1)
    return DefectPatchDataset(
        items=items,
        patch_size=manifest.get("patch_size", 512),
        is_train=is_train,
        patches_per_image=patches_per_image,
        num_classes=num_classes,
        **augment,
    )
