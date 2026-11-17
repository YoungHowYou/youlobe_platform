# -*- coding: utf-8 -*-
"""多分类语义分割（背景 + N 个缺陷类）回归测试。

覆盖：数据集类别标签、CE+Dice 损失、混淆矩阵指标、1 epoch 训练闭环、
多分类滑窗推理与类别图。

    .venv/bin/python -m unittest tests.test_semseg_multiclass -v
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from core.semseg import build_model
from core.semseg.infer import connected_components_by_class, predict_class_map
from core.semseg.losses import CEDiceLoss, confusion_from_pred, metrics_from_confusion
from core.semseg.trainer import run_semseg_train

NUM_CLASSES = 4          # 0=背景, 1..3 = 三类缺陷
SIZE = 160
PATCH = 128


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    buf.tofile(str(path))


def _make_split(root: Path, stems: list[str]) -> list[dict]:
    items = []
    for i, stem in enumerate(stems):
        img = np.full((SIZE, SIZE), 60 + i * 10, np.uint8)
        ids = np.zeros((SIZE, SIZE), np.uint8)
        # 每张图放两个不同类别的方块（类别在 1..3 轮换）
        c1 = 1 + (i % 3)
        c2 = 1 + ((i + 1) % 3)
        ids[30:70, 30:70] = c1
        ids[90:130, 90:130] = c2
        img[30:70, 30:70] = 200
        img[90:130, 90:130] = 220
        binary = np.where(ids > 0, 255, 0).astype(np.uint8)

        _write_png(root / "images" / f"{stem}.png", img)
        _write_png(root / "masks_binary" / f"{stem}.png", binary)
        _write_png(root / "masks_ids" / f"{stem}.png", ids)
        ys, xs = np.where(binary > 0)
        pts = np.stack([ys, xs], axis=1).astype(np.int32)
        (root / "points").mkdir(parents=True, exist_ok=True)
        np.save(root / "points" / f"{stem}.npy", pts)

        items.append({
            "stem": stem,
            "image": str(root / "images" / f"{stem}.png"),
            "mask": str(root / "masks_binary" / f"{stem}.png"),
            "mask_ids": str(root / "masks_ids" / f"{stem}.png"),
            "points": str(root / "points" / f"{stem}.npy"),
        })
    return items


class SemsegMulticlassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        train = _make_split(cls.root, [f"t{i}" for i in range(3)])
        val = _make_split(cls.root, [f"v{i}" for i in range(2)])
        cls.manifest_path = cls.root / "semseg_dataset" / "manifest.json"
        cls.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        cls.manifest_path.write_text(json.dumps({
            "version": 1,
            "task_type": "semseg",
            "mode": "multiclass",
            "patch_size": PATCH,
            "num_classes": NUM_CLASSES,
            "class_names": ["隐裂", "崩边", "缺口"],
            "include_ok": False,
            "train": train,
            "val": val,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ---- 数据集 ----
    def test_dataset_returns_class_labels(self):
        from core.semseg.dataset import build_datasets

        ds = build_datasets(str(self.manifest_path), is_train=True,
                            patches_per_image=2, num_classes=NUM_CLASSES)
        sample = ds[0]
        self.assertEqual(tuple(sample["image"].shape), (3, PATCH, PATCH))
        mask = sample["mask"]
        self.assertEqual(mask.dtype, torch.int64)          # 类别索引，不是 0/255
        self.assertEqual(tuple(mask.shape), (PATCH, PATCH))
        self.assertTrue(set(int(v) for v in mask.unique()) <= set(range(NUM_CLASSES)))

    def test_binary_dataset_unchanged(self):
        from core.semseg.dataset import build_datasets

        ds = build_datasets(str(self.manifest_path), is_train=False,
                            patches_per_image=1, num_classes=1)
        sample = ds[0]
        self.assertEqual(tuple(sample["mask"].shape), (1, PATCH, PATCH))
        self.assertTrue(set(np.unique(sample["mask"].numpy())).issubset({0.0, 255.0}))

    # ---- 损失与指标 ----
    def test_cedice_loss_and_metrics(self):
        model = build_model(num_classes=NUM_CLASSES, pretrained=False).eval()
        x = torch.zeros(1, 3, PATCH, PATCH)
        with torch.no_grad():
            logits = model(x)
        label = torch.zeros(1, PATCH, PATCH, dtype=torch.long)
        label[:, 20:60, 20:60] = 2
        loss = CEDiceLoss(num_classes=NUM_CLASSES)(logits, label)
        self.assertTrue(torch.isfinite(loss))

        # 用确定性的 pred/label 验证指标定义（不用随机权重的 argmax）
        gt = torch.zeros(1, PATCH, PATCH, dtype=torch.long)
        gt[:, 20:60, 20:60] = 2
        pred = gt.clone()
        conf = confusion_from_pred(pred, gt, NUM_CLASSES)
        self.assertEqual(tuple(conf.shape), (NUM_CLASSES, NUM_CLASSES))
        metrics = metrics_from_confusion(conf)
        self.assertAlmostEqual(metrics["dice"], 1.0, places=6)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=6)
        # class_dice[0] 对应 pixel_value=1，class_dice[1] 对应 pixel_value=2
        self.assertAlmostEqual(metrics["class_dice"][1], 1.0, places=6)
        # 3 个前景类里只有第 2 类出现，剩余两类按 0 计入平均
        self.assertAlmostEqual(metrics["mdice"], 1.0 / 3, places=6)

    def test_ignore_index_pixels_are_skipped(self):
        label = torch.zeros(1, 10, 10, dtype=torch.long)
        label[:, :5, :5] = 1
        label[:, 5:, 5:] = 255            # 忽略区
        pred = torch.zeros(1, 10, 10, dtype=torch.long)
        conf = confusion_from_pred(pred, label, 2)
        self.assertEqual(float(conf.sum()), 75.0)   # 100 - 25 个忽略像素

    # ---- 推理 ----
    def test_predict_class_map(self):
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        img = np.full((SIZE, SIZE), 128, np.uint8)
        cls_map, fg = predict_class_map(model, img, num_classes=NUM_CLASSES,
                                       window=PATCH, stride=96, threshold=0.5,
                                       device="cpu")
        self.assertEqual(cls_map.shape, (SIZE, SIZE))
        self.assertEqual(cls_map.dtype, np.uint8)
        self.assertEqual(fg.shape, (SIZE, SIZE))
        self.assertTrue(set(np.unique(cls_map)) <= set(range(NUM_CLASSES)))

        # 强制一张已知类别图，验证逐类连通域
        fake = np.zeros((SIZE, SIZE), np.uint8)
        fake[10:40, 10:40] = 1
        fake[80:120, 80:120] = 3
        comps = connected_components_by_class(fake, min_area=10)
        self.assertEqual(sorted({c["pixel_value"] for c in comps}), [1, 3])
        self.assertEqual(sorted({c["class_index"] for c in comps}), [0, 2])

    def test_export_wrapper_probability(self):
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        x = torch.zeros(1, 3, PATCH, PATCH)
        prob = model.forward_prob(x)
        self.assertEqual(tuple(prob.shape), (1, NUM_CLASSES, PATCH, PATCH))
        self.assertAlmostEqual(float(prob.sum(dim=1).mean()), 1.0, places=4)

    # ---- 训练闭环 ----
    def test_train_one_epoch_multiclass(self):
        out = self.root / "runs_mc"
        result = run_semseg_train(
            manifest_path=str(self.manifest_path),
            output_dir=str(out),
            epochs=1,
            patch_size=PATCH,
            patches_per_image=1,
            batch_size=1,
            accum_steps=1,
            lr=1e-4,
            workers=0,
            device="cpu",
            pretrained=False,
            num_classes=NUM_CLASSES,
            fg_class_weight=20.0,
            class_names=["隐裂", "崩边", "缺口"],
        )
        self.assertTrue(Path(result["best_pt"]).exists())
        self.assertTrue(Path(result["last_pt"]).exists())

        # results.csv 必须有 mIoU / mDice 两列
        with open(result["results_csv"], encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        self.assertEqual(len(rows), 1)
        for col in ("dice", "iou", "recall", "miou", "mdice", "train_loss", "val_loss"):
            self.assertIn(col, rows[0])
            self.assertNotEqual(rows[0][col], "")

        # checkpoint meta 记录类别信息，供推理/导出/继承使用
        ckpt = torch.load(result["best_pt"], map_location="cpu", weights_only=False)
        meta = ckpt.get("meta") or {}
        self.assertEqual(meta.get("num_classes"), NUM_CLASSES)
        self.assertEqual(meta.get("mode"), "multiclass")
        self.assertEqual(meta.get("class_names"), ["隐裂", "崩边", "缺口"])
        self.assertEqual(ckpt["model"]["decode_head.conv_seg.weight"].shape[0], NUM_CLASSES)


if __name__ == "__main__":
    unittest.main()
