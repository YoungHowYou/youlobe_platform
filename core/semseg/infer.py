# -*- coding: utf-8 -*-
"""语义分割推理：滑窗 + 平均 + 阈值。

**口径必须与** ``scratch_seg/tools/evaluate.py::sliding_window_predict``
**逐行一致**，否则复现不出实验里的 Dice：

1. 输入按灰度读取 → 复制成 3 通道 → ``[:, :, ::-1]``（实验的 BGR→RGB 写法）
2. 归一化：``(x - [123.675,116.28,103.53]) / [58.395,57.12,57.375]``（0~255 输入）
3. 小图右下补 0 到 ``window``
4. 步长自适应：大图用 ``stride``；一旦图不满足 ``H >= window + stride``，
   退化为 ``max(1, (H - window)//2)``（640 图因此会变成 64 步长、9 个窗口）
5. 模型输出 stride-2 logits → **双线性上采样到 window** → sigmoid
6. 重叠区域**算术平均**（累加 / 计数），不是取最大
7. 默认阈值 0.6（实验阈值扫描最优点）
"""

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

DEFAULT_WINDOW = 512
DEFAULT_STRIDE = 384
DEFAULT_THRESHOLD = 0.6


def resolve_device(requested: str | torch.device | None = "auto") -> torch.device:
    """把平台/Ultralytics 风格的设备串归一化成 ``torch.device``。

    平台训练配置里 ``device`` 沿用的是 Ultralytics 约定：
    ``"0"`` / ``"cuda"`` / ``"cuda:0"`` / ``"cpu"`` / ``"auto"``。
    注意 ``torch.device("0")`` 是**非法**的，必须转成 ``cuda:0``。
    """
    if isinstance(requested, torch.device):
        return requested
    text = str(requested or "auto").strip().lower()
    cuda_ok = torch.cuda.is_available()

    if text in ("", "auto", "cuda", "gpu"):
        return torch.device("cuda:0" if cuda_ok else "cpu")
    if text == "cpu":
        return torch.device("cpu")
    if text.startswith("cuda:"):
        return torch.device(text) if cuda_ok else torch.device("cpu")
    if text.isdigit():
        return torch.device(f"cuda:{text}") if cuda_ok else torch.device("cpu")
    if text.startswith("gpu:"):
        idx = text.split(":", 1)[1] or "0"
        return torch.device(f"cuda:{idx}") if cuda_ok else torch.device("cpu")
    return torch.device(text)


def read_image_gray(path: str | Path) -> Optional[np.ndarray]:
    """读取灰度图，支持中文路径（Windows 下 cv2.imread 会失败）。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def read_image_bgr(path: str | Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def preprocess_plane(img_gray: np.ndarray) -> np.ndarray:
    """灰度 (H,W) → 归一化后的 (H,W,3) float32（RGB 顺序）。"""
    img3 = np.repeat(img_gray[:, :, None], 3, axis=2).astype(np.float32)
    img3 = (img3[:, :, ::-1] - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img3, dtype=np.float32)


def window_starts(size: int, window: int, stride: int) -> range:
    """与实验一致的窗口起点序列。调用前必须保证 ``size >= window``。"""
    if size >= window + stride:
        step = stride
    else:
        step = max(1, (size - window) // 2)
    return range(0, size - window + 1, step)


def _tile_starts(size: int, window: int, step: int) -> list[int]:
    """Halo 分块的起点序列：等步长 + 强制最后一个窗口贴右/下边。

    只用中心有效区累加时，最后一个窗口必须恰好让有效区结束在图像边界，
    否则右侧/底部会留一条没被覆盖的缝。
    """
    step = max(1, int(step))
    last = max(0, size - window)
    starts = list(range(0, last + 1, step))
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _sliding_window_map(
    model: nn.Module,
    img_gray: np.ndarray,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    device: str | torch.device | None = "auto",
    amp: bool = False,
    num_classes: int = 1,
    valid_size: int | None = None,
) -> np.ndarray:
    """灰度图 → 概率图。二分类返回 (H,W)，多分类返回 (C,H,W)（C=num_classes）。

    **Halo**：``window`` 是送给网络的输入边长，``valid_size`` 是要保留的中心有效区。
    两者差值的两倍即 halo（四周各扩 halo 像素上下文）。有效区小于窗口时，
    图像先四周补 halo（与训练 pad 口径一致），滑窗只累加中心有效区，
    这样块边界处的预测不会污染结果；不传 ``valid_size`` 时退化为原行为。
    """
    if img_gray is None:
        raise ValueError("输入图像为空")
    if img_gray.ndim == 3:
        # 兼容误传 BGR：直接转灰度，与训练/实验的灰度口径保持一致
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    n_cls = max(1, int(num_classes))
    valid = int(valid_size or window)
    if valid > window:
        raise ValueError(f"valid_size({valid}) 不能大于 window({window})")
    halo = (int(window) - valid) // 2
    if halo > 0:
        img_gray = cv2.copyMakeBorder(img_gray, halo, halo, halo, halo,
                                      cv2.BORDER_CONSTANT, value=0)

    dev = resolve_device(device)
    model = model.to(dev)
    was_training = model.training
    model.eval()

    plane = preprocess_plane(img_gray)
    h0, w0 = img_gray.shape[:2]        # 含 halo 的画布尺寸
    h, w = h0, w0
    if h < window or w < window:
        plane = cv2.copyMakeBorder(
            plane, 0, max(0, window - h), 0, max(0, window - w),
            cv2.BORDER_CONSTANT, value=0,
        )
        h, w = plane.shape[:2]

    acc = np.zeros((n_cls, h, w), np.float32)
    count = np.zeros((h, w), np.float32)

    if halo > 0:
        step = max(1, min(int(stride), valid))
        ys = _tile_starts(h, window, step)
        xs = _tile_starts(w, window, step)
    else:
        ys = window_starts(h, window, stride)
        xs = window_starts(w, window, stride)

    for y in ys:
        for x in xs:
            patch = plane[y:y + window, x:x + window]
            tensor = torch.from_numpy(
                np.ascontiguousarray(patch.transpose(2, 0, 1))
            ).unsqueeze(0).to(dev)
            with torch.no_grad():
                if amp and dev.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(tensor)
                else:
                    logits = model(tensor)
                logits = F.interpolate(
                    logits.float(), size=(window, window),
                    mode="bilinear", align_corners=False,
                )
                probs = (torch.sigmoid(logits) if n_cls == 1
                         else torch.softmax(logits, dim=1))
                p = probs[0].float().cpu().numpy()
            if n_cls == 1 and p.ndim == 2:        # (win,win) → (1,win,win)
                p = p[None]
            # 只取中心有效区（Halo 分块的核心）
            ty, tx = y + halo, x + halo
            acc[:, ty:ty + valid, tx:tx + valid] += p[:, halo:halo + valid,
                                                      halo:halo + valid]
            count[ty:ty + valid, tx:tx + valid] += 1.0

    # 裁掉 halo 补边 + 右侧/底部的小图 pad，回到原图尺寸
    out = acc[:, halo:halo + h0 - 2 * halo, halo:halo + w0 - 2 * halo] / \
        np.maximum(count[halo:halo + h0 - 2 * halo, halo:halo + w0 - 2 * halo], 1.0)[None]
    if was_training:
        model.train()
    return out[0].astype(np.float32) if n_cls == 1 else out.astype(np.float32)


def sliding_window_predict(
    model: nn.Module,
    img_gray: np.ndarray,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    device: str | torch.device | None = "auto",
    amp: bool = False,
    valid_size: int | None = None,
) -> np.ndarray:
    """灰度图 (H,W) → 前景概率图 (H,W) float32，滑窗平均（可选 Halo 有效区）。"""
    return _sliding_window_map(model, img_gray, window=window, stride=stride,
                               device=device, amp=amp, num_classes=1,
                               valid_size=valid_size)


def sliding_window_predict_multi(
    model: nn.Module,
    img_gray: np.ndarray,
    num_classes: int,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    device: str | torch.device | None = "auto",
    amp: bool = False,
    valid_size: int | None = None,
) -> np.ndarray:
    """灰度图 → 多分类概率图 (C,H,W)，第 0 通道是背景（可选 Halo 有效区）。"""
    return _sliding_window_map(model, img_gray, window=window, stride=stride,
                               device=device, amp=amp, num_classes=num_classes,
                               valid_size=valid_size)


def predict_prob(
    model: nn.Module,
    image: str | Path | np.ndarray,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    device: str | torch.device | None = "auto",
    amp: bool = False,
    valid_size: int | None = None,
) -> np.ndarray:
    """路径或数组 → 前景概率图。路径按灰度读取（与实验一致）。"""
    img = _load_gray(image)
    return sliding_window_predict(model, img, window=window, stride=stride,
                                  device=device, amp=amp, valid_size=valid_size)


def predict_prob_multi(
    model: nn.Module,
    image: str | Path | np.ndarray,
    num_classes: int,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    device: str | torch.device | None = "auto",
    amp: bool = False,
    valid_size: int | None = None,
) -> np.ndarray:
    """路径或数组 → 多分类概率图 (C,H,W)，第 0 通道是背景。"""
    img = _load_gray(image)
    return sliding_window_predict_multi(model, img, num_classes=num_classes,
                                        window=window, stride=stride,
                                        device=device, amp=amp,
                                        valid_size=valid_size)


def predict_class_map(
    model: nn.Module,
    image: str | Path | np.ndarray,
    num_classes: int,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    threshold: float = DEFAULT_THRESHOLD,
    device: str | torch.device | None = "auto",
    amp: bool = False,
    valid_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """路径或数组 → (类别图 uint8, 前景概率 float32)。

    类别图像素值 = class_index + 1（0 = 背景），与平台 ``masks_ids`` 同口径。
    前景判定：`max(P_c, c>=1) > threshold`（对照实验的合并二分类口径）。
    """
    probs = predict_prob_multi(model, image, num_classes=num_classes,
                               window=window, stride=stride,
                               device=device, amp=amp, valid_size=valid_size)
    fg = probs[1:].max(axis=0) if probs.shape[0] > 1 else probs[0]
    cls = (probs[1:].argmax(axis=0).astype(np.uint8) + 1) if probs.shape[0] > 1 \
        else np.zeros(fg.shape, np.uint8)
    cls = np.where(fg > float(threshold), cls, 0).astype(np.uint8)
    return cls, fg.astype(np.float32)


def _load_gray(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = read_image_gray(image)
        if img is None:
            raise ValueError(f"图像读取失败：{image}")
        return img
    return image


def predict_mask(
    model: nn.Module,
    image: str | Path | np.ndarray,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    threshold: float = DEFAULT_THRESHOLD,
    device: str | torch.device | None = "auto",
    amp: bool = False,
) -> np.ndarray:
    """路径或数组 → 二值 mask（uint8，0/255）。"""
    prob = predict_prob(model, image, window=window, stride=stride,
                        device=device, amp=amp)
    return (prob > float(threshold)).astype(np.uint8) * 255


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    """写 PNG，支持中文路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"PNG 编码失败：{path}")
    buf.tofile(str(path))


def connected_components(mask: np.ndarray, min_area: int = 0) -> list[dict]:
    """二值 mask → 连通域列表（面积 / bbox / 质心），供前端缺陷统计与小图裁剪。"""
    binary = (mask > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rows = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx].tolist()
        if area < min_area:
            continue
        rows.append({
            "label": idx,
            "area": int(area),
            "bbox": [int(x), int(y), int(w), int(h)],
            "centroid": [float(centroids[idx][0]), float(centroids[idx][1])],
        })
    return rows


def connected_components_by_class(class_map: np.ndarray, min_area: int = 0) -> list[dict]:
    """多分类类别图 → 逐类连通域，附带 ``pixel_value``（类别图值）与 class_index。"""
    rows: list[dict] = []
    if class_map is None or class_map.size == 0:
        return rows
    for value in range(1, int(class_map.max()) + 1):
        sub = np.where(class_map == value, 255, 0).astype(np.uint8)
        for comp in connected_components(sub, min_area=min_area):
            comp["pixel_value"] = int(value)
            comp["class_index"] = int(value) - 1     # 平台 class_index 是 0 基
            rows.append(comp)
    return rows
