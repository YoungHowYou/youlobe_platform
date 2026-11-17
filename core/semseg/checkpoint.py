# -*- coding: utf-8 -*-
"""Checkpoint 读写与权重加载。

存在两个来源：

1. **MMSeg checkpoint**（实验产出，如 ``best_mIoU_iter_2000.pth``）：
   顶层是 ``{'meta':..., 'state_dict': {...}, 'message_hub':..., ...}``。
   ``meta`` / ``message_hub`` 里 pickle 了 ``mmengine`` 的对象（如
   ``HistoryBuffer``），而平台不安装 mmengine。这里用一个**只在本进程内生效**
   的兼容反序列化器把这些对象替换成占位对象——``state_dict`` 里的张量仍由
   torch 自己恢复，不受影响。

2. **平台 checkpoint**：``{'model': state_dict, 'meta': {...}}``，
   由 :func:`save_platform_checkpoint` 写出。

两条路径最终都归一化成「键名 = 模型模块名」的 state_dict，配合
``core/semseg/model.py`` 中与 MMSeg 完全一致的模块命名，可以直接
``model.load_state_dict(..., strict=True)``。
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# mmengine 兼容反序列化
# --------------------------------------------------------------------------
class _StubValue:
    """占位对象：可调用、可继续取属性，用来吞掉 pickle 里的非张量对象。"""

    def __call__(self, *args, **kwargs):        # noqa: D102
        return self

    def __getattr__(self, item):                # noqa: D105
        return self

    def __setstate__(self, state):              # noqa: D105
        self.__dict__["_state"] = state


class _StubMeta(type):
    """让占位**类**本身也能响应任意属性（pickle 会 getattr(HistoryBuffer, 'min')）。"""

    def __getattr__(cls, item):                 # noqa: D105
        return _StubValue()


class _Stub(metaclass=_StubMeta):
    def __new__(cls, *args, **kwargs):          # noqa: D102
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):        # noqa: D107
        pass

    def __setstate__(self, state):              # noqa: D105
        self.__dict__["_state"] = state


_STUB_ROOTS = ("mmengine", "mmcv", "mmseg", "mmdet")


def _make_stub_class(module: str, name: str) -> type:
    return _StubMeta("Stub", (_Stub,), {"__module__": module})


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in _STUB_ROOTS:
            return _make_stub_class(module, name)
        return super().find_class(module, name)


class _CompatPickleModule:
    """模拟 ``pickle`` 模块接口，torch.load 只需要这些属性。"""

    Unpickler = _CompatUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """读取 checkpoint，兼容 MMSeg（含 mmengine 对象）与平台格式。

    只用于**本机可信文件**：``weights_only=False`` 会执行 pickle 反序列化。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")

    try:
        ckpt = torch.load(
            str(path),
            map_location=map_location,
            weights_only=False,
            pickle_module=_CompatPickleModule,
        )
    except Exception as exc:
        # 少数 checkpoint 是纯张量字典，用安全模式再试一次
        try:
            ckpt = torch.load(str(path), map_location=map_location, weights_only=True)
        except Exception:
            raise RuntimeError(f"无法读取 checkpoint {path}：{exc}") from exc
    return ckpt


def extract_state_dict(ckpt: Any, prefer_ema: bool = False) -> Dict[str, torch.Tensor]:
    """从各种 checkpoint 包装中取出 state_dict。"""
    if not isinstance(ckpt, dict):
        raise TypeError(f"不支持的 checkpoint 类型：{type(ckpt)}")

    if prefer_ema and isinstance(ckpt.get("ema"), dict):
        return ckpt["ema"]
    for key in ("state_dict", "model", "model_state_dict", "weights"):
        value = ckpt.get(key)
        if isinstance(value, dict) and value:
            return value
    return ckpt


def _strip_wrapper_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """去掉 ``module.`` / ``model.`` / ``_orig_mod.`` 这类训练包装前缀。"""
    prefixes = ("module.", "model.", "_orig_mod.")
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        out[new_key] = value
    return out


def load_weights(
    model: nn.Module,
    checkpoint: str | Path | Dict[str, torch.Tensor],
    strict: bool = True,
    prefer_ema: bool = False,
    verbose: bool = True,
    skip_mismatch: bool = False,
) -> Tuple[list[str], list[str]]:
    """把权重加载进模型，返回 ``(missing_keys, unexpected_keys)``。

    ``strict=True`` 时任何键不匹配都会抛错——这正是我们要的：M1 阶段必须
    证明平台实现与实验 checkpoint 逐键一致。

    ``skip_mismatch=True`` 时把**形状不一致**的权重（例如二值 head 的
    ``conv_seg`` vs 多分类 head 的 ``conv_seg``）当 missing 跳过，
    用于「二值 ⇄ 多分类」或类别数变动时的继承训练。
    """
    if isinstance(checkpoint, (str, Path)):
        state_dict = _strip_wrapper_prefix(
            extract_state_dict(load_checkpoint(checkpoint), prefer_ema=prefer_ema)
        )
    else:
        state_dict = _strip_wrapper_prefix(checkpoint)

    if skip_mismatch:
        target = model.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        dropped: list[str] = []
        for key, value in state_dict.items():
            expect = target.get(key)
            if expect is not None and tuple(expect.shape) != tuple(value.shape):
                dropped.append(key)
                continue
            filtered[key] = value
        if dropped and verbose:
            print(f"[semseg] 形状不匹配已跳过 {len(dropped)} 项：{dropped[:5]}")
        state_dict = filtered

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if verbose:
        print(f"[semseg] 权重加载：{len(state_dict)} 个张量，"
              f"missing={len(missing)}，unexpected={len(unexpected)}")
        if missing:
            print(f"[semseg] missing 前 10 项：{missing[:10]}")
        if unexpected:
            print(f"[semseg] unexpected 前 10 项：{unexpected[:10]}")
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"权重与模型不完全匹配：missing={len(missing)}，unexpected={len(unexpected)}"
        )
    return list(missing), list(unexpected)


def read_checkpoint_meta(checkpoint: str | Path | dict) -> dict:
    """读取 meta（MMSeg 记录 iter/epoch/config 快照），非 dict 时返回空。"""
    ckpt = load_checkpoint(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else None
    return meta if isinstance(meta, dict) else {}


# --------------------------------------------------------------------------
# 写入
# --------------------------------------------------------------------------
def save_platform_checkpoint(model: nn.Module, path: str | Path,
                             meta: Optional[dict] = None) -> str:
    """保存为平台格式 ``{'model': ..., 'meta': ...}``。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta or {}}, str(path))
    return str(path)


def find_checkpoint(directory: str | Path, prefer_best: bool = True) -> Optional[str]:
    """在实验目录里挑一个 checkpoint：优先 ``best_mIoU_iter_*.pth``，其次 ``iter_*.pth``。"""
    directory = Path(directory)
    if directory.is_file():
        return str(directory)
    if not directory.is_dir():
        return None
    patterns = ["best_mIoU_iter_*.pth", "best*.pth"] if prefer_best else []
    patterns += ["iter_*.pth", "*.pth"]
    for pattern in patterns:
        candidates = sorted(directory.glob(pattern))
        if candidates:
            return str(candidates[-1])
    return None
