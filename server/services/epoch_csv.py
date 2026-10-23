# -*- coding: utf-8 -*-
"""
从 `results.csv` 补训练集 loss
==============================
背景：`train_epoch_logs` 里的 `train_*_loss` 曾经一直是 NULL——旧版
`core/train.py` 把 ultralytics 的 `trainer.loss_items` / `trainer.tloss`
当成 Tensor 处理，而它们从 8.3 起是 `{loss名: Tensor}` 的 dict，
异常被 `except` 吞掉 → 前端 Loss 曲线只有验证集（Val）一条线。

现在回调已修好，但**已经在跑的旧进程**仍会继续写 NULL。这个模块在读取
epoch 日志时兜底：拿到任务目录下 ultralytics 的 `results.csv`，把缺失的
`train_*_loss` 按 epoch 补上（只补 None，绝不覆盖库里已有的值）。

对齐规则：`results.csv` 的 `epoch` 从 1 开始（`save_metrics` 写
`self.epoch + 1`），而库里用的是 `trainer.epoch`（0 基），所以统一 -1。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# results.csv 列名 → train_epoch_logs 属性名
_TRAIN_LOSS_COLS = {
    "train/box_loss": "train_box_loss",
    "train/seg_loss": "train_seg_loss",
    "train/cls_loss": "train_cls_loss",
    "train/dfl_loss": "train_dfl_loss",
    "train/loss": "train_cls_loss",     # cls 任务只有一个 loss
}
_ATTRS = (
    "train_box_loss",
    "train_seg_loss",
    "train_cls_loss",
    "train_dfl_loss",
)

# {csv_path: (mtime, {epoch: {attr: value}})} —— 训练中 CSV 每轮都会变，
# 所以按 mtime 失效，避免每次请求都重新解析整个文件。
_CACHE: Dict[str, tuple[float, Dict[int, Dict[str, float]]]] = {}
_CACHE_LIMIT = 32


def _find_results_csv(output_dir: Optional[str]) -> Optional[Path]:
    if not output_dir:
        return None
    base = Path(output_dir)
    if not base.is_dir():
        return None
    hits = sorted(base.rglob("results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def _to_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # CSV 里的 nan/inf 不能原样回传：既会让前端曲线断开，也会让 JSON 序列化失败
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _load_csv(csv_path: Path) -> Dict[int, Dict[str, float]]:
    try:
        mtime = csv_path.stat().st_mtime
    except OSError:
        return {}
    cached = _CACHE.get(str(csv_path))
    if cached and cached[0] == mtime:
        return cached[1]

    parsed: Dict[int, Dict[str, float]] = {}
    try:
        with csv_path.open(encoding="utf-8", errors="ignore", newline="") as fh:
            for raw in csv.DictReader(fh):
                row = {(k or "").strip(): v for k, v in raw.items()}
                ep = _to_float(row.get("epoch"))
                if ep is None:
                    continue
                fields: Dict[str, float] = {}
                for col, attr in _TRAIN_LOSS_COLS.items():
                    if attr in fields:      # 同名字段（cls 的 train/loss）先到先得
                        continue
                    val = _to_float(row.get(col))
                    if val is not None:
                        fields[attr] = val
                if fields:
                    parsed[int(ep) - 1] = fields
    except OSError:
        return {}

    if len(_CACHE) > _CACHE_LIMIT:
        _CACHE.clear()
    _CACHE[str(csv_path)] = (mtime, parsed)
    return parsed


def fill_missing_train_loss(output_dir: Optional[str], logs: Iterable[Any]) -> int:
    """给 EpochLogOut 列表补训练集 loss，返回实际补上的字段数。

    `logs` 里每个元素需要有 `.epoch` 属性且允许 setattr（pydantic 模型即可）。
    """
    rows_by_epoch = None
    filled = 0
    for log in logs:
        if all(getattr(log, attr, None) is None for attr in _ATTRS):
            if rows_by_epoch is None:
                csv_path = _find_results_csv(output_dir)
                rows_by_epoch = _load_csv(csv_path) if csv_path else {}
                if not rows_by_epoch:
                    return 0
            data = rows_by_epoch.get(int(log.epoch))
            if not data:
                continue
            for attr, val in data.items():
                setattr(log, attr, val)
                filled += 1
    return filled
