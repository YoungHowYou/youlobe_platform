# -*- coding: utf-8 -*-
"""语义分割 ONNX FP16 导出回归测试（ResUNet34 专用）。

``torch.onnx.export`` 只产出 FP32 图，FP16 需要在导出后做一次图转换。
本测试锁住三件事，防止再出现“记录写 FP16、文件其实是 FP32”的假精度：

  - 产物名 / metadata 指向 ``semseg_{imgsz}_fp16.onnx``，precision=FP16
  - 图内权重是 float16，但**输入输出仍为 float32**（部署端预处理不用改）
  - 转换后图能通过 onnx.checker，且 onnxruntime 推理结果与 FP32 一致

    .venv/bin/python -m unittest tests.test_semseg_onnx_fp16 -v
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from core.semseg import build_model, save_platform_checkpoint
from core.semseg.export import _calibration_input, export_semseg

HAS_ORT = importlib.util.find_spec("onnxruntime") is not None

NUM_CLASSES = 3          # 多分类：输出 softmax 概率 (1,N,H,W)
IMGSZ = 128


def _make_patch(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.random((IMGSZ, IMGSZ)) * 60 + 80).astype(np.uint8)
    img[30:70, 30:70] = 225
    return img


class SemsegOnnxFp16ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        torch.manual_seed(2026)
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        # 放大权重 + 给前景类加偏置，避免输出几乎全背景使对比过于平凡
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(3.0)
            model.decode_head.conv_seg.bias[1] = 3.0
        cls.ckpt = save_platform_checkpoint(
            model, cls.root / "m.pt",
            meta={"num_classes": NUM_CLASSES, "class_names": ["a", "b"],
                  "mode": "multiclass", "patch_size": IMGSZ},
        )
        cls.patch = _make_patch()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _meta(self, path: Path) -> dict:
        lines = (path / "metadata.yaml").read_text(encoding="utf-8").splitlines()
        return dict(
            (k.strip(), v.strip())
            for k, _, v in (ln.partition(":") for ln in lines if ":" in ln)
        )

    def test_fp16_artifacts_and_metadata(self):
        import onnx

        out = export_semseg(
            self.ckpt, str(self.root / "f16"), export_format="onnx",
            imgsz=IMGSZ, half=True, num_classes=NUM_CLASSES,
            class_names=["a", "b"],
        )
        self.assertEqual(out["precision"], "FP16")
        fp16_path = Path(out["export_path"])
        self.assertTrue(fp16_path.exists())
        self.assertEqual(fp16_path.name, f"semseg_{IMGSZ}_fp16.onnx")
        self.assertEqual(Path(out["fp16_path"]), fp16_path)
        # 中间 FP32 图仍在（INT8 路径同样保留），但对外产物是 FP16
        self.assertTrue((Path(out["onnx_path"])).exists())
        self.assertEqual(Path(out["onnx_path"]).name, f"semseg_{IMGSZ}.onnx")

        meta = self._meta(self.root / "f16")
        self.assertEqual(meta["precision"], "FP16")
        self.assertEqual(meta["model_file"], fp16_path.name)

        graph = onnx.load(str(fp16_path)).graph
        dtypes = {}
        for t in graph.initializer:
            dtypes[t.data_type] = dtypes.get(t.data_type, 0) + 1
        self.assertGreater(
            dtypes.get(onnx.TensorProto.FLOAT16, 0), 0, f"未见 FP16 权重: {dtypes}"
        )
        # 权重转成 FP16 后不该再残留 float32 初始化张量
        self.assertEqual(dtypes.get(onnx.TensorProto.FLOAT, 0), 0, f"残留 FP32 权重: {dtypes}")

        # I/O 必须是 float32，部署端继续按 metadata（0~255 NCHW float32）喂数据
        for vi in list(graph.input) + list(graph.output):
            self.assertEqual(
                vi.type.tensor_type.elem_type, onnx.TensorProto.FLOAT,
                f"{vi.name} 不是 float32",
            )
        onnx.checker.check_model(str(fp16_path))      # 拓扑序必须合法

    def test_fp32_unaffected(self):
        out = export_semseg(
            self.ckpt, str(self.root / "f32"), export_format="onnx",
            imgsz=IMGSZ, num_classes=NUM_CLASSES, class_names=["a", "b"],
        )
        self.assertEqual(out["precision"], "FP32")
        self.assertNotIn("fp16_path", out)
        self.assertEqual(Path(out["export_path"]).name, f"semseg_{IMGSZ}.onnx")
        self.assertEqual(self._meta(self.root / "f32")["precision"], "FP32")

    @unittest.skipUnless(HAS_ORT, "需要 onnxruntime")
    def test_fp16_matches_fp32(self):
        out32 = export_semseg(
            self.ckpt, str(self.root / "cmp32"), export_format="onnx",
            imgsz=IMGSZ, num_classes=NUM_CLASSES, class_names=["a", "b"],
        )
        out16 = export_semseg(
            self.ckpt, str(self.root / "cmp16"), export_format="onnx",
            imgsz=IMGSZ, half=True, num_classes=NUM_CLASSES,
            class_names=["a", "b"],
        )
        import onnxruntime as ort

        x = _calibration_input(self.patch, IMGSZ)
        def run(path: str):
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            return sess.run(None, {"input": x})[0]

        a, b = run(out32["export_path"]), run(out16["export_path"])
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(b.dtype, np.float32)            # 输出仍是 float32
        diff = np.abs(a - b)
        # 随机权重且被人为放大 3 倍，属于误差很大的极端情形：实测 max≈7.7e-3、
        # mean≈4.4e-4，这里按 2 倍余量卡住，真实训练权重只会更小。
        self.assertLess(float(diff.max()), 2e-2, f"max|Δ|={diff.max():.3e}")
        self.assertLess(float(diff.mean()), 2e-3, f"mean|Δ|={diff.mean():.3e}")
        self.assertGreater(
            float((a.argmax(0) == b.argmax(0)).mean()), 0.999,
            "FP16 与 FP32 的 argmax 一致率过低",
        )
        # FP16 权重体积应明显小于 FP32
        self.assertLess(
            Path(out16["export_path"]).stat().st_size,
            Path(out32["export_path"]).stat().st_size,
        )


if __name__ == "__main__":
    unittest.main()
