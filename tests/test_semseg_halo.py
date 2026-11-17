# -*- coding: utf-8 -*-
"""语义分割 Halo（输入带上下文 / 只在中心有效区算 loss 与滑窗）回归测试。

    .venv/bin/python -m unittest tests.test_semseg_halo -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from core.semseg import build_model
from core.semseg.dataset import (
    DefectPatchDataset, build_datasets, probe_image_sizes, resolve_halo_for_manifest,
)
from core.semseg.infer import predict_prob, sliding_window_predict
from core.semseg.trainer import run_semseg_train

VALID = 96          # 有效区
HALO = 16           # 四周各扩 16 → 输入 128
SIZE = 192


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    buf.tofile(str(path))


def _make_items(root: Path, stems: list[str]) -> list[dict]:
    items = []
    for i, stem in enumerate(stems):
        img = np.full((SIZE, SIZE), 90 + i * 10, np.uint8)
        ids = np.zeros((SIZE, SIZE), np.uint8)
        ids[60:120, 60:120] = 1
        img[60:120, 60:120] = 210
        binary = np.where(ids > 0, 255, 0).astype(np.uint8)
        _write_png(root / "images" / f"{stem}.png", img)
        _write_png(root / "masks_binary" / f"{stem}.png", binary)
        _write_png(root / "masks_ids" / f"{stem}.png", ids)
        ys, xs = np.where(binary > 0)
        (root / "points").mkdir(parents=True, exist_ok=True)
        np.save(root / "points" / f"{stem}.npy", np.stack([ys, xs], 1).astype(np.int32))
        items.append({
            "stem": stem,
            "image": str(root / "images" / f"{stem}.png"),
            "mask": str(root / "masks_binary" / f"{stem}.png"),
            "mask_ids": str(root / "masks_ids" / f"{stem}.png"),
            "points": str(root / "points" / f"{stem}.npy"),
        })
    return items


class SemsegHaloTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        train = _make_items(cls.root, [f"t{i}" for i in range(3)])
        val = _make_items(cls.root, [f"v{i}" for i in range(2)])
        cls.manifest = cls.root / "semseg_dataset" / "manifest.json"
        cls.manifest.parent.mkdir(parents=True, exist_ok=True)
        cls.manifest.write_text(json.dumps({
            "version": 1, "task_type": "semseg", "mode": "binary",
            "patch_size": VALID, "num_classes": 1, "class_names": ["defect"],
            "include_ok": False, "train": train, "val": val,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ---- 数据集：输入含 halo，标签只保留有效区 ----
    def test_dataset_shapes_with_halo(self):
        ds = build_datasets(str(self.manifest), is_train=True,
                            patches_per_image=1, halo=HALO)
        sample = ds[0]
        self.assertEqual(ds.valid_size, VALID)
        self.assertEqual(ds.halo, HALO)
        self.assertEqual(tuple(sample["image"].shape), (3, VALID + 2 * HALO, VALID + 2 * HALO))
        # 二分类：mask 是 (1, valid, valid)
        self.assertEqual(tuple(sample["mask"].shape), (1, VALID, VALID))

    def test_dataset_multiclass_mask_is_valid_only(self):
        ds = build_datasets(str(self.manifest), is_train=True, patches_per_image=1,
                            num_classes=2, halo=HALO)
        sample = ds[0]
        self.assertEqual(tuple(sample["image"].shape), (3, VALID + 2 * HALO, VALID + 2 * HALO))
        self.assertEqual(tuple(sample["mask"].shape), (VALID, VALID))

    def test_halo_zero_keeps_old_behavior(self):
        ds = build_datasets(str(self.manifest), is_train=True, patches_per_image=1)
        sample = ds[0]
        self.assertEqual(ds.halo, 0)
        self.assertEqual(tuple(sample["image"].shape), (3, VALID, VALID))
        self.assertEqual(tuple(sample["mask"].shape), (1, VALID, VALID))

    def test_odd_halo_is_rounded_down_to_even(self):
        ds = build_datasets(str(self.manifest), is_train=True, patches_per_image=1,
                            halo=17)
        self.assertEqual(ds.halo, 16)     # logits 是 stride-2，必须偶数

    # ---- 整图即一个 patch（图像不大于有效区）时 halo 自动关闭 ----
    def test_halo_auto_disabled_when_image_fits_patch(self):
        items = _make_items(self.root / "small", ["s0"])          # 192×192
        ds = DefectPatchDataset(items, patch_size=SIZE, is_train=True, halo=HALO)
        self.assertEqual(ds.halo_requested, HALO)
        self.assertEqual(ds.halo, 0)
        self.assertEqual(ds.input_size, SIZE)
        self.assertIn("自动关闭", ds.halo_note)
        self.assertEqual(tuple(ds[0]["image"].shape), (3, SIZE, SIZE))

    def test_halo_kept_when_image_bigger_than_patch(self):
        items = _make_items(self.root / "big", ["b0"])            # 192×192
        ds = DefectPatchDataset(items, patch_size=VALID, is_train=True, halo=HALO)
        self.assertEqual(ds.halo, HALO)
        self.assertEqual(ds.input_size, VALID + 2 * HALO)
        self.assertEqual(ds.halo_note, "")

    def test_probe_prefers_manifest_size_and_skips_file_read(self):
        items = [{"image": "/nonexistent/x.png", "width": 640, "height": 480}]
        self.assertEqual(probe_image_sizes(items), (640, 480))

    def test_resolve_halo_for_manifest_matches_dataset(self):
        # manifest 的 patch 尺寸 = 图像尺寸 → 整图即一个 patch → 关闭
        halo, note = resolve_halo_for_manifest(str(self.manifest), SIZE, HALO)
        self.assertEqual(halo, 0)
        self.assertIn("自动关闭", note)
        halo, note = resolve_halo_for_manifest(str(self.manifest), VALID, HALO)
        self.assertEqual(halo, HALO)
        self.assertEqual(note, "")

    # ---- 推理：只累加中心有效区，整幅图都要被覆盖 ----
    def test_sliding_window_halo_covers_whole_image(self):
        model = build_model(num_classes=1, pretrained=False)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.decode_head.conv_seg.bias[0] = 30.0      # 恒输出前景

        img = np.full((SIZE, SIZE), 120, np.uint8)
        prob = sliding_window_predict(model, img, window=VALID + 2 * HALO,
                                     stride=VALID // 2, device="cpu",
                                     valid_size=VALID)
        self.assertEqual(prob.shape, (SIZE, SIZE))
        # 每个像素都被覆盖到（没被覆盖的位置概率会是 0）
        self.assertGreater(float(prob.min()), 0.9)

    def test_predict_prob_halo_shape(self):
        model = build_model(num_classes=1, pretrained=False)
        img = np.zeros((SIZE, SIZE), np.uint8)
        prob = predict_prob(model, img, window=VALID + 2 * HALO, stride=VALID // 2,
                            device="cpu", valid_size=VALID)
        self.assertEqual(prob.shape, (SIZE, SIZE))
        self.assertTrue(np.isfinite(prob).all())

    # ---- 训练：halo 训练能跑通且 meta 记录口径 ----
    def test_train_with_halo_records_meta(self):
        out = self.root / "runs_halo"
        res = run_semseg_train(
            manifest_path=str(self.manifest), output_dir=str(out),
            epochs=1, patch_size=VALID, patches_per_image=1, batch_size=1,
            accum_steps=1, workers=0, device="cpu", pretrained=False,
            num_classes=1, halo=HALO,
        )
        ckpt = torch.load(res["best_pt"], map_location="cpu", weights_only=False)
        meta = ckpt["meta"]
        self.assertEqual(meta["halo"], HALO)
        self.assertEqual(meta["patch_size"], VALID)
        self.assertEqual(meta["input_size"], VALID + 2 * HALO)


if __name__ == "__main__":
    unittest.main()
