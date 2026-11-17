# -*- coding: utf-8 -*-
"""语义分割 INT8 量化导出回归测试（NNCF PTQ + OpenVINO/ONNX）。

只验证「能不能量化、量化后还对不对」，不追求真实精度（模型是随机权重）：
  - 缺校准 patch 必须报错（防止拿噪声/全背景凑数）
  - INT8 产物 meta 记录 precision/quantization/calibration_patches
  - INT8 与 FP32 推理结果高度一致（前景 IoU / argmax 一致率）

    .venv/bin/python -m unittest tests.test_semseg_int8 -v
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from core.semseg import build_model, save_platform_checkpoint
from core.semseg.export import _calibration_input, export_semseg

HAS_OV = importlib.util.find_spec("openvino") is not None
HAS_NNCF = importlib.util.find_spec("nncf") is not None
HAS_ORT = importlib.util.find_spec("onnxruntime") is not None

NUM_CLASSES = 3
IMGSZ = 128


def _make_patches(count: int = 4, seed: int = 0) -> list:
    """造几张「有亮缺陷块」的灰度 patch，模拟真实缺陷样本。"""
    rng = np.random.default_rng(seed)
    patches = []
    for _ in range(count):
        img = (rng.random((IMGSZ, IMGSZ)) * 60 + 80).astype(np.uint8)
        y, x = rng.integers(20, IMGSZ - 40, size=2)
        img[y:y + 30, x:x + 30] = 230
        patches.append(img)
    return patches


class SemsegInt8ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        torch.manual_seed(1234)      # 量化前后一致性对权重分布敏感，固定种子保证可复现
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        # 放大权重 + 给第 1 类加偏置，避免输出几乎全背景导致量化对比过于平凡
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(3.0)
            model.decode_head.conv_seg.bias[1] = 3.0
        cls.ckpt = save_platform_checkpoint(
            model, cls.root / "m.pt",
            meta={"num_classes": NUM_CLASSES, "class_names": ["a", "b"],
                  "mode": "multiclass"},
        )
        cls.patches = _make_patches()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _meta(self, path: Path) -> dict:
        lines = (path / "metadata.yaml").read_text(encoding="utf-8").splitlines()
        return dict(
            (k.strip(), v.strip())
            for k, _, v in (ln.partition(":") for ln in lines if ":" in ln)
        )

    def test_int8_requires_real_patches(self):
        with self.assertRaises(RuntimeError) as ctx:
            export_semseg(self.ckpt, str(self.root / "no_calib"),
                          export_format="openvino", imgsz=IMGSZ, int8=True,
                          num_classes=NUM_CLASSES)
        self.assertIn("缺陷", str(ctx.exception))

    def test_int8_fp16_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            export_semseg(self.ckpt, str(self.root / "bad"),
                          export_format="openvino", imgsz=IMGSZ,
                          int8=True, half=True,
                          calibration_patches=self.patches,
                          num_classes=NUM_CLASSES)

    @unittest.skipUnless(HAS_OV and HAS_NNCF, "需要 openvino + nncf")
    def test_openvino_int8_parity(self):
        out32 = export_semseg(self.ckpt, str(self.root / "ov_fp32"),
                              export_format="openvino", imgsz=IMGSZ,
                              num_classes=NUM_CLASSES,
                              class_names=["a", "b"])
        out8 = export_semseg(
            self.ckpt, str(self.root / "ov_int8"), export_format="openvino",
            imgsz=IMGSZ, int8=True, calibration_patches=self.patches,
            calibration_subset=len(self.patches),
            num_classes=NUM_CLASSES, class_names=["a", "b"],
        )
        self.assertEqual(out8["precision"], "INT8")
        meta8 = self._meta(self.root / "ov_int8")
        self.assertEqual(meta8["precision"], "INT8")
        self.assertEqual(meta8["quantization"], "nncf_ptq")
        self.assertEqual(int(meta8["calibration_patches"]), len(self.patches))
        self.assertEqual(self._meta(self.root / "ov_fp32")["precision"], "FP32")

        fp32_bin = Path(out32["xml_path"]).with_suffix(".bin").stat().st_size
        int8_bin = Path(out8["xml_path"]).with_suffix(".bin").stat().st_size
        self.assertLess(int8_bin, fp32_bin)      # 量化后权重应明显变小

        import openvino as ov

        core = ov.Core()

        def infer(xml_path: str, arr):
            model = core.read_model(xml_path)
            compiled = core.compile_model(model, "CPU")
            return list(compiled({0: arr}).values())[0][0]

        ious, agrees = [], []
        for patch in self.patches:
            x = _calibration_input(patch, IMGSZ)
            a = infer(out32["xml_path"], x)
            b = infer(out8["xml_path"], x)
            fa, fb = a[1:].max(0) > 0.5, b[1:].max(0) > 0.5
            union = (fa | fb).sum()
            ious.append(1.0 if union == 0 else float((fa & fb).sum() / union))
            agrees.append(float((a.argmax(0) == b.argmax(0)).mean()))
        # 随机权重 + 小样本（4 patch）的 PTQ 会产生一定量化噪声，阈值按实测留余量
        self.assertGreater(float(np.mean(ious)), 0.8, f"前景 IoU 过低: {ious}")
        self.assertGreater(float(np.mean(agrees)), 0.88, f"argmax 一致率过低: {agrees}")

    @unittest.skipUnless(HAS_NNCF and HAS_ORT, "需要 nncf + onnxruntime")
    def test_onnx_int8(self):
        out = export_semseg(
            self.ckpt, str(self.root / "onnx_int8"), export_format="onnx",
            imgsz=IMGSZ, int8=True, calibration_patches=self.patches,
            calibration_subset=len(self.patches), num_classes=NUM_CLASSES,
        )
        self.assertEqual(out["precision"], "INT8")
        int8_path = Path(out["export_path"])
        self.assertTrue(int8_path.exists())
        self.assertTrue(int8_path.name.endswith("_int8.onnx"))
        # fp32 与 int8 都远小于 fp32 基线（这里只验证能跑起来、输出形状对）
        import onnxruntime as ort

        sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input": _calibration_input(self.patches[0], IMGSZ)})[0]
        self.assertEqual(y.shape, (1, NUM_CLASSES, IMGSZ, IMGSZ))
        self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
