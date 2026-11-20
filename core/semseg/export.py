# -*- coding: utf-8 -*-
"""语义分割模型导出：ONNX / OpenVINO（FP32 / FP16 / INT8）。

导出图与实验 ``export_onnx.py`` 完全一致：

    input (1,3,H,W) 0~255 归一化后的张量
      → backbone + decode_head（stride-2 logits）
      → 双线性上采样到 H,W
      → sigmoid（多分类是 softmax）
      → output (1,C,H,W) 概率图

**INT8**（2026-11-16 支持）：用 NNCF 做 Post-Training Quantization（PTQ），
校准数据必须是**真实缺陷 patch**——缺陷只占 0.004%，用随机噪声/随机裁剪校准
会让 quantizer 把缺陷区间压掉（实测 Dice 会从 0.7 掉到 0.01 量级）。
因此 ``export_semseg`` 要求调用方传入 ``calibration_patches``（灰度 patch 列表），
缺省时直接报错而不是悄悄用噪声凑数。

**FP16**：OpenVINO 用 ``compress_to_fp16`` 压缩权重；ONNX 侧用
``_convert_onnx_fp16()`` 做一次图转换（权重 FP16、输入输出仍 FP32，
部署端预处理不用改）。只有 GPU 上的 onnxruntime EP 才会明显变快，
CPU 上 FP16 无提速，要 CPU 低延迟请用 INT8。
"""

from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from .checkpoint import load_weights
from .infer import DEFAULT_STRIDE, DEFAULT_THRESHOLD, DEFAULT_WINDOW
from .model import build_model


class ExportWrapper(nn.Module):
    """上采样 + sigmoid/softmax，输出与输入同分辨率的概率图。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_prob(x)


def _precision_label(half: bool, int8: bool) -> str:
    if int8:
        return "INT8"
    return "FP16" if half else "FP32"


def _write_metadata(output_dir: Path, imgsz: int, half: bool,
                    window: int, stride: int, threshold: float,
                    num_classes: int = 1,
                    class_names: Optional[list] = None,
                    int8: bool = False,
                    calibration_count: int = 0,
                    model_file: str = "",
                    valid_size: int = 0) -> Path:
    """写 metadata.yaml：推理端据此决定预处理/窗口/阈值/类别/精度。"""
    path = output_dir / "metadata.yaml"
    multi = int(num_classes) > 1
    names = [str(n) for n in (class_names or [])]
    valid = int(valid_size or imgsz)
    halo = max(0, (int(imgsz) - valid) // 2)
    lines = [
        "task: semseg",
        "arch: resunet34_cldice",
        f"mode: {'multiclass' if multi else 'binary'}",
        f"num_classes: {int(num_classes)}",
        f"imgsz: {int(imgsz)}",
        f"valid: {valid}",
        f"halo: {halo}",
        f"precision: {_precision_label(half, int8)}",
    ]
    if model_file:
        lines.append(f"model_file: {model_file}")
    lines += [
        f"window: {int(window)}",
        f"stride: {int(stride)}",
        f"threshold: {float(threshold)}",
        "input_layout: NCHW",
        "input_range: 0-255",
        "mean: [123.675, 116.28, 103.53]",
        "std: [58.395, 57.12, 57.375]",
        "bgr_to_rgb: true",
        f"output: {'softmax_probability_channels' if multi else 'sigmoid_probability'}",
    ]
    if int8:
        lines.append("quantization: nncf_ptq")
        lines.append(f"calibration_patches: {int(calibration_count)}")
    if multi:
        # 0 = 背景；1..N 对应项目 class_index 0..N-1（与 masks_ids 像素值一致）
        lines.append("class_names: [" + ", ".join(f'"{n}"' for n in names) + "]")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _calibration_input(patch, imgsz: int):
    """灰度/BGR patch → 与推理完全一致的 (1,3,H,W) float32 输入。"""
    import cv2
    import numpy as np

    from .infer import preprocess_plane

    arr = np.asarray(patch)
    if arr is None or arr.size == 0:
        raise ValueError("校准 patch 为空")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    h, w = arr.shape[:2]
    if h < imgsz or w < imgsz:
        arr = cv2.copyMakeBorder(arr, 0, max(0, imgsz - h), 0, max(0, imgsz - w),
                                 cv2.BORDER_CONSTANT, value=0)
        h, w = arr.shape[:2]
    if (h, w) != (imgsz, imgsz):        # 过大则中心裁剪（与训练 patch 口径一致）
        y0, x0 = (h - imgsz) // 2, (w - imgsz) // 2
        arr = arr[y0:y0 + imgsz, x0:x0 + imgsz]
    plane = preprocess_plane(arr)
    return plane.transpose(2, 0, 1)[None].astype(np.float32)


def _make_calibration_dataset(patches: Sequence, imgsz: int,
                              onnx_inputs: bool = False):
    """NNCF 校准数据集：逐条吐 (1,3,imgsz,imgsz) float32。

    OpenVINO 后端接受 ndarray 本身；ONNX 后端（onnxruntime）要求
    ``{"input": array}`` 形式的 dict，因此按后端切换。
    """
    import nncf

    def _iter():
        for patch in patches:
            arr = _calibration_input(patch, imgsz)
            yield {"input": arr} if onnx_inputs else arr

    return nncf.Dataset(_iter())


def _quantize(model, calibration, subset_size: int):
    """NNCF PTQ 统一入口（兼容 2.x 的 ModelPreset 与 3.x 的 QuantizationPreset）。"""
    import nncf

    preset_cls = (getattr(nncf, "ModelPreset", None)
                  or getattr(nncf, "QuantizationPreset", None))
    kwargs: dict = {"subset_size": max(1, int(subset_size))}
    if preset_cls is not None:
        kwargs["preset"] = preset_cls.PERFORMANCE
    return nncf.quantize(model, calibration, **kwargs)


def _toposort_graph(graph) -> None:
    """就地按拓扑序重排节点（Kahn 算法）。

    onnxruntime 的 float16 转换在 ``keep_io_types=True`` 时会把**输入 Cast
    追加到图末尾**，导致 ``onnx.checker`` 报 “Nodes in a graph must be
    topologically sorted”。这里按真实依赖重新排一次，不改变任何节点内容。
    """
    from collections import deque

    nodes = list(graph.node)
    n = len(nodes)
    producer: dict = {}
    for idx, node in enumerate(nodes):
        for out in node.output:
            if out:
                producer[out] = idx

    external = {vi.name for vi in graph.input}
    external.update(t.name for t in graph.initializer)
    indeg = [0] * n
    consumers: list = [[] for _ in range(n)]
    for idx, node in enumerate(nodes):
        for inp in node.input:
            if not inp or inp in external:
                continue
            src = producer.get(inp)
            if src is None or src == idx:
                continue
            indeg[idx] += 1
            consumers[src].append(idx)

    queue = deque(i for i, d in enumerate(indeg) if d == 0)
    order = []
    while queue:
        idx = queue.popleft()
        order.append(idx)
        for c in consumers[idx]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)

    if len(order) != n:      # 存在环/外部输入异常：保持原序，交给 checker 报错
        return
    del graph.node[:]
    graph.node.extend(nodes[i] for i in order)


def _convert_onnx_fp16(src: Path, dst: Path) -> Path:
    """把 ONNX 图的算子权重/计算转成 FP16，**输入输出仍保持 FP32**。

    ``keep_io_types=True`` 会在图首尾插入 Cast，部署端依旧按 metadata
    （NCHW、0~255、float32）喂数据，预处理无需改动；只有 GPU 上的
    onnxruntime（CUDA/TensorRT EP）才能拿到实际加速，CPU EP 不会变快。

    转换器优先级：onnxruntime 自带的 ``transformers.float16``（已随
    onnxruntime 安装）→ ``onnxconverter-common``（可选替代）。
    """
    import onnx

    try:
        from onnxruntime.transformers.float16 import (
            convert_float_to_float16 as _convert,
        )
    except Exception:                                    # pragma: no cover - 环境相关
        try:
            from onnxconverter_common.float16 import (
                convert_float_to_float16 as _convert,
            )
        except Exception as exc:                          # pragma: no cover - 环境相关
            raise RuntimeError(
                "ONNX FP16 转换需要 onnxruntime（自带 float16 转换器）或 "
                "onnxconverter-common，请先执行 `.venv/bin/pip install onnxruntime`。"
            ) from exc

    model = onnx.load(str(src))
    fp16_model = _convert(model, keep_io_types=True)
    _toposort_graph(fp16_model.graph)

    # 自检 1：I/O 必须还是 FP32（否则部署端喂 float32 会直接报 dtype 不匹配）
    io_types = {vi.name: vi.type.tensor_type.elem_type for vi in fp16_model.graph.input}
    io_types.update({vi.name: vi.type.tensor_type.elem_type for vi in fp16_model.graph.output})
    bad_io = [k for k, v in io_types.items() if v != onnx.TensorProto.FLOAT]
    if bad_io:
        raise RuntimeError(
            "ONNX FP16 转换后输入/输出不再是 float32（"
            + ", ".join(bad_io) + "），已中止，避免导出不可用的模型。"
        )

    onnx.save(fp16_model, str(dst))

    # 自检 2：落盘后必须是合法 ONNX 图
    try:
        onnx.checker.check_model(str(dst))
    except Exception as exc:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX FP16 转换结果未通过 onnx.checker：{exc}") from exc
    return dst


def export_semseg(
    model_path: str,
    output_dir: str,
    export_format: str = "onnx",
    imgsz: int = 512,
    half: bool = False,
    int8: bool = False,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    threshold: float = DEFAULT_THRESHOLD,
    num_classes: int = 1,
    class_names: Optional[list] = None,
    calibration_patches: Optional[Sequence] = None,
    calibration_subset: int = 64,
    valid_size: int = 0,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """导出语义分割模型，返回 ``{format, onnx_path, export_path, imgsz, precision}``。

    Args:
        int8: 是否做 NNCF PTQ 量化（仅 OpenVINO / ONNX 产物，FP16 与 INT8 互斥）
        calibration_patches: **真实缺陷 patch**（灰度 ndarray 列表），INT8 必填；
            由调用方（导出路由）从项目标注里按缺陷中心裁出，避免用随机裁剪校准
        calibration_subset: 参与量化的 patch 数上限
    """
    fmt = str(export_format or "onnx").lower()
    if fmt not in ("onnx", "openvino"):
        raise ValueError(f"语义分割暂只支持 onnx / openvino，收到 {export_format}")
    if int8 and half:
        raise ValueError("INT8 与 FP16 互斥，请只选一种精度")

    imgsz = int(imgsz or 512)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if int8:
        try:
            import nncf  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "语义分割 INT8 需要 nncf：请在平台 venv 执行 "
                "`.venv/bin/pip install nncf`（OpenVINO 后端）后重试。"
            ) from exc
        patches = [p for p in (calibration_patches or []) if p is not None]
        if not patches:
            raise RuntimeError(
                "INT8 量化必须提供**真实缺陷 patch** 做校准（随机/全背景 patch 会把"
                "缺陷概率区间压掉，实测 Dice 会崩到 0.01 量级）。"
                "请确认项目里已有标注好的缺陷图，或先导出 FP16。"
            )
        patches = patches[:max(1, int(calibration_subset))]

    if progress_callback:
        progress_callback(10, 100, "加载模型...")
    # 类别数 / 有效区 / halo 优先以 checkpoint 自带的 meta 为准（训练口径跟着权重走）
    valid = int(valid_size or 0)
    try:
        from .checkpoint import read_checkpoint_meta
        meta = read_checkpoint_meta(model_path)
        num_classes = int(meta.get("num_classes") or num_classes or 1)
        if meta.get("patch_size"):
            valid = int(meta["patch_size"])
        if meta.get("halo") is not None:
            meta_halo = max(0, int(meta.get("halo") or 0))
            meta_input = valid + 2 * meta_halo
            if meta_input != int(imgsz):
                print(f"[semseg 导出] 按 checkpoint meta 调正输入尺寸：{imgsz} → {meta_input}"
                      f"（有效区 {valid} + 2×halo {meta_halo}）")
            imgsz = meta_input
        if class_names is None and meta.get("class_names"):
            class_names = list(meta["class_names"])
    except Exception as exc:
        print(f"[semseg 导出] 读取 checkpoint meta 失败，按传入类别数 {num_classes} 导出：{exc}")
    valid = int(valid or imgsz)
    # 文件名跟随**调正后**的真实输入尺寸，保证与 metadata.yaml 的 imgsz 一致
    stem = f"semseg_{imgsz}"
    model = build_model(num_classes=int(num_classes), pretrained=False)
    load_weights(model, model_path, strict=False, verbose=False, skip_mismatch=True)
    model.eval()
    wrapper = ExportWrapper(model).eval()

    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)
    onnx_path = output / f"{stem}.onnx"

    if progress_callback:
        progress_callback(30, 100, "导出 ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(onnx_path),
            opset_version=17,
            input_names=["input"], output_names=["output"],
            dynamic_axes=None,
        )

    calib_count = len(patches) if int8 else 0
    # INT8/FP16 用带后缀的文件名，避免覆盖同任务之前导出的其他精度
    ir_name = f"{stem}_int8.xml" if int8 else (f"{stem}_fp16.xml" if half else f"{stem}.xml")
    if fmt == "openvino":
        model_file = ir_name
    else:
        model_file = (f"{stem}_int8.onnx" if int8
                      else (f"{stem}_fp16.onnx" if half else f"{stem}.onnx"))

    def _meta() -> None:
        """metadata 只在产物真正生成后写，避免失败时留下“声明 FP16 实为 FP32”的残留。"""
        _write_metadata(output, imgsz, half, window, stride, threshold,
                        num_classes=int(num_classes), class_names=class_names,
                        int8=bool(int8), calibration_count=calib_count,
                        model_file=model_file, valid_size=valid)

    # ---------- ONNX ----------
    if fmt == "onnx":
        export_path = onnx_path
        fp16_path = None
        if int8:
            if progress_callback:
                progress_callback(60, 100, "NNCF 量化 ONNX（INT8）...")
            import onnx

            calib = _make_calibration_dataset(patches, imgsz, onnx_inputs=True)
            int8_model = _quantize(onnx.load(str(onnx_path)), calib, calib_count)
            export_path = output / f"{stem}_int8.onnx"
            onnx.save(int8_model, str(export_path))
        elif half:
            # ResUNet34 不是 ultralytics 图模型，torch.onnx.export 只产出 FP32 图，
            # FP16 必须显式做一次图转换（I/O 仍为 FP32）。
            if progress_callback:
                progress_callback(60, 100, "转换 ONNX FP16...")
            fp16_path = _convert_onnx_fp16(onnx_path, output / f"{stem}_fp16.onnx")
            export_path = fp16_path

        _meta()
        if progress_callback:
            progress_callback(100, 100, "导出完成")
        result = {
            "format": "onnx",
            "onnx_path": str(onnx_path),
            "export_path": str(export_path),
            "imgsz": imgsz,
            "precision": _precision_label(half, int8),
        }
        if int8:
            result["int8_path"] = str(export_path)
            result["calibration_patches"] = calib_count
        if fp16_path is not None:
            result["fp16_path"] = str(fp16_path)
        return result

    # ---------- OpenVINO ----------
    if progress_callback:
        progress_callback(70, 100, "转换为 OpenVINO IR...")
    import openvino as ov

    ov_model = ov.convert_model(str(onnx_path))
    xml_path = output / ir_name

    if int8:
        if progress_callback:
            progress_callback(85, 100, f"NNCF 量化 OpenVINO（INT8，{calib_count} 张缺陷 patch）...")

        calib = _make_calibration_dataset(patches, imgsz)
        quantized = _quantize(ov_model, calib, calib_count)
        # INT8 必须关闭 fp16 压缩，否则权重会被再次压成 FP16
        ov.save_model(quantized, str(xml_path), compress_to_fp16=False)
    else:
        ov.save_model(ov_model, str(xml_path), compress_to_fp16=bool(half))

    _meta()
    if progress_callback:
        progress_callback(100, 100, "导出完成")
    return {
        "format": "openvino",
        "onnx_path": str(onnx_path),
        "export_path": str(output),
        "xml_path": str(xml_path),
        "imgsz": imgsz,
        "precision": _precision_label(half, int8),
        "calibration_patches": calib_count,
    }
