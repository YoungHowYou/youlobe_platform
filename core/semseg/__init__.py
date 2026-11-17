# -*- coding: utf-8 -*-
"""语义分割（玻璃缺陷）训练与推理模块。

本包是 scratch_seg 实验（MMSegmentation 1.2.2）的**纯 PyTorch 复刻**，
不依赖 mmengine / mmcv / mmseg，避免在平台 venv 里编译 mmcv。

对外 API：
    build_model              构造 ResUNet34 + FCNHead
    load_checkpoint          读取 MMSeg / 平台两种 checkpoint 格式
    load_weights             把权重灌进模型并返回 missing/unexpected 报告
    BCEDiceClDiceLoss        实验最优 Loss（BCE(pos_weight=5)+Dice+clDice）
    sliding_window_predict   滑窗概率图（与 scratch_seg/tools/evaluate.py 同口径）
    predict_mask             二值 mask
"""

from .checkpoint import (
    extract_state_dict,
    find_checkpoint,
    load_checkpoint,
    load_weights,
    save_platform_checkpoint,
)
from .infer import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    connected_components_by_class,
    predict_class_map,
    predict_mask,
    predict_prob,
    predict_prob_multi,
    read_image_gray,
    sliding_window_predict,
    sliding_window_predict_multi,
)
from .losses import (
    BCEDiceClDiceLoss,
    BCEDiceLoss,
    CEDiceLoss,
    build_criterion,
    metrics_from_confusion,
    soft_skeleton,
)
from .metrics import evaluate_pair, skeletonize, summarize
from .model import FCNHead, ResUNet34Backbone, ResUNet34Segmenter, build_model

__all__ = [
    "build_model",
    "ResUNet34Segmenter",
    "ResUNet34Backbone",
    "FCNHead",
    "load_checkpoint",
    "extract_state_dict",
    "load_weights",
    "save_platform_checkpoint",
    "find_checkpoint",
    "BCEDiceClDiceLoss",
    "BCEDiceLoss",
    "CEDiceLoss",
    "build_criterion",
    "metrics_from_confusion",
    "soft_skeleton",
    "evaluate_pair",
    "skeletonize",
    "summarize",
    "sliding_window_predict",
    "sliding_window_predict_multi",
    "predict_class_map",
    "predict_prob_multi",
    "connected_components_by_class",
    "predict_mask",
    "predict_prob",
    "read_image_gray",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
