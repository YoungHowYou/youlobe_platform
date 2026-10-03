# -*- coding: utf-8 -*-
"""
GPU 训练链路自检
================
用 4 张合成小图跑 1 个 epoch，验证 `core/train.py` → Ultralytics → torch CUDA 全链路可用。

- 不触碰 `storage/` 里的真实项目数据，产物落在系统临时目录
- 会按需下载 `yolo11n-seg.pt`（约 6MB，需要联网；离线机器请先把预训练权重放到项目根目录）
- 日志里出现 `[train device] 自动选择 GPU: cuda:0（...）` 即说明设备选择走到了 GPU

用法：
    venv\\Scripts\\python.exe tools\\check_gpu_train.py
    venv\\Scripts\\python.exe tools\\check_gpu_train.py --device cpu     # 只验证 CPU 路径
"""

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_fake_dataset(root: Path, n: int = 4, size: int = 128) -> str:
    """生成 n 张随机图 + 同一个多边形标注，images/{train,val} + labels/{train,val}。"""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    poly = "0.30 0.30 0.70 0.30 0.70 0.70 0.30 0.70"
    for i in range(n):
        split = "val" if i == n - 1 else "train"
        img = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(root / "images" / split / f"{i:03d}.png"), img)
        (root / "labels" / split / f"{i:03d}.txt").write_text(f"0 {poly}\n", encoding="utf-8")

    from core.train import generate_dataset_yaml
    return generate_dataset_yaml(str(root), str(root / "dataset.yaml"), class_names=["defect"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto / cpu / 0 / cuda:0")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    args = parser.parse_args()

    import torch
    from core.train import resolve_device, run_train

    print("=" * 60)
    print(f"torch {torch.__version__}  cuda_build={torch.version.cuda}  "
          f"available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"resolve_device({args.device!r}) -> {resolve_device(args.device)!r}")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="gpu_train_check_") as tmp:
        tmp_path = Path(tmp)
        dataset_yaml = build_fake_dataset(tmp_path / "dataset")
        try:
            run_train(
                dataset_yaml=dataset_yaml,
                output_dir=str(tmp_path / "runs"),
                model_name=args.model,
                imgsz=128,
                epochs=1,
                batch_size=2,
                patience=0,
                device=args.device,
                task_type="seg",
                close_mosaic=0,
                workers=0,
            )
        except Exception as exc:  # noqa: BLE001
            print("\n[FAIL] 训练链路异常：")
            traceback.print_exc()
            return 1

    print("\n[OK] 训练链路可用（1 epoch 完成）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
