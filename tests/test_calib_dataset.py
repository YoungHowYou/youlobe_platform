# -*- coding: utf-8 -*-
"""INT8 校准集解析回归测试。

    .venv/bin/python -m unittest tests.test_calib_dataset -v
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from server.services.calib_dataset import (
    _label_line, ensure_calib_dataset, make_calib_from_dataset,
)


def _png(path: Path, size=(32, 32), value=128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", np.full((size[1], size[0]), value, np.uint8))
    assert ok
    buf.tofile(str(path))


class CalibDatasetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    # ---- 从训练数据集抽样 ----
    def test_make_calib_from_det_dataset(self):
        ds = self.root / "dataset"
        for i in range(3):
            _png(ds / "images" / "train" / f"a{i}.png")
            (ds / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (ds / "labels" / "train" / f"a{i}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        _png(ds / "images" / "val" / "v0.png")
        (ds / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (ds / "labels" / "val" / "v0.txt").write_text("0 0.4 0.4 0.1 0.1\n",
                                                     encoding="utf-8")

        out = make_calib_from_dataset(ds, "det", ["缺陷"], self.root / "calib", limit=2)
        self.assertIsNotNone(out)
        self.assertTrue((out / "dataset.yaml").is_file())
        yml = (out / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("0: 缺陷", yml)
        self.assertIn("train: images", yml)
        imgs = sorted((out / "images").glob("*.png"))
        self.assertEqual(len(imgs), 2)
        self.assertEqual(len(sorted((out / "labels").glob("*.txt"))), 2)

    def test_make_calib_writes_empty_label_for_unlabeled_image(self):
        ds = self.root / "dataset"
        _png(ds / "images" / "train" / "only_img.png")
        out = make_calib_from_dataset(ds, "seg", ["a"], self.root / "calib", limit=4)
        self.assertIsNotNone(out)
        lbl = out / "labels" / "0000.txt"
        self.assertTrue(lbl.is_file())
        self.assertEqual(lbl.read_text(encoding="utf-8"), "")

    def test_make_calib_from_cls_dataset(self):
        ds = self.root / "dataset"
        for cls in ("0000", "0001"):
            for i in range(3):
                _png(ds / cls / f"{cls}_{i}.png")
        out = make_calib_from_dataset(ds, "cls", ["裂纹", "崩边"],
                                      self.root / "calib", limit=4)
        self.assertIsNotNone(out)
        self.assertTrue((out / "train" / "0000").is_dir())
        self.assertTrue((out / "val" / "0001").is_dir())
        yml = (out / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("train: train", yml)
        self.assertIn("0: 裂纹", yml)

    def test_make_calib_returns_none_for_missing_dataset(self):
        self.assertIsNone(
            make_calib_from_dataset(self.root / "nope", "det", [], self.root / "calib"))

    # ---- 解析优先级 ----
    def test_ensure_prefers_training_dataset(self):
        out_root = self.root / "task_1"
        (out_root / "dataset").mkdir(parents=True)
        (out_root / "dataset" / "dataset.yaml").write_text("path: x\n", encoding="utf-8")
        (out_root / "calib").mkdir(parents=True)
        (out_root / "calib" / "dataset.yaml").write_text("path: y\n", encoding="utf-8")
        task = SimpleNamespace(output_dir=str(out_root), project_id=1, config={})
        path, note = ensure_calib_dataset(task, None, None)
        self.assertTrue(path.endswith("dataset"))
        self.assertIn("训练数据集", note)

    def test_ensure_falls_back_to_kept_calib(self):
        out_root = self.root / "task_2"
        (out_root / "calib").mkdir(parents=True)
        (out_root / "calib" / "dataset.yaml").write_text("path: y\n", encoding="utf-8")
        task = SimpleNamespace(output_dir=str(out_root), project_id=1, config={})
        path, note = ensure_calib_dataset(task, None, None)
        self.assertTrue(path.endswith("calib"))
        self.assertIn("校准集", note)

    def test_ensure_reports_reason_without_project(self):
        out_root = self.root / "task_3"
        out_root.mkdir(parents=True)
        task = SimpleNamespace(output_dir=str(out_root), project_id=1, config={})
        path, note = ensure_calib_dataset(task, None, None)
        self.assertIsNone(path)
        self.assertIn("项目", note)

    def test_keep_calib_for_task_is_idempotent(self):
        from server.services.calib_dataset import keep_calib_for_task

        out_root = self.root / "task_9"
        ds = out_root / "dataset"
        _png(ds / "images" / "train" / "a.png")
        (ds / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (ds / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n",
                                                     encoding="utf-8")
        task = SimpleNamespace(id=9, output_dir=str(out_root), project_id=1,
                               config={"class_names": ["x"]})
        made = keep_calib_for_task(task, None)      # db=None → 走 task_type 默认 seg
        self.assertIsNotNone(made)
        self.assertTrue((out_root / "calib" / "dataset.yaml").is_file())
        self.assertEqual(keep_calib_for_task(task, None), made)   # 已存在则复用

    # ---- 标签换算 ----
    def test_label_line_det_and_seg(self):
        pts = [{"x": 0.2, "y": 0.2}, {"x": 0.6, "y": 0.2},
               {"x": 0.6, "y": 0.8}, {"x": 0.2, "y": 0.8}]
        det = _label_line("det", 1, pts, 100, 100)
        parts = [float(v) for v in det.split()]
        self.assertEqual(parts[0], 1.0)
        self.assertAlmostEqual(parts[1], 0.4, places=5)   # cx
        self.assertAlmostEqual(parts[2], 0.5, places=5)   # cy
        self.assertAlmostEqual(parts[3], 0.4, places=5)   # w
        self.assertAlmostEqual(parts[4], 0.6, places=5)   # h
        seg = _label_line("seg", 0, pts, 100, 100)
        self.assertEqual(len(seg.split()), 1 + 8)
        self.assertIsNone(_label_line("det", 0, pts[:2], 100, 100))   # 点太少


if __name__ == "__main__":
    unittest.main()
