# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class SegLossParams(BaseModel):
    """实例分割掩膜损失的超参（只有对应损失类型会用到其中一部分）。"""
    dice_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="bce_dice：Dice 占比")
    focal_gamma: float = Field(default=2.0, ge=0.0, le=10.0, description="focal：难例聚焦强度")
    focal_alpha: float = Field(default=0.25, ge=0.0, le=1.0, description="focal：正样本权重")
    tversky_alpha: float = Field(default=0.3, ge=0.0, le=1.0, description="tversky：FP 惩罚权重")
    tversky_beta: float = Field(default=0.7, ge=0.0, le=1.0, description="tversky：FN 惩罚权重")
    focal_tversky_gamma: float = Field(default=0.75, ge=0.0, le=5.0, description="focal_tversky：难例放大指数")
    boundary_weight: float = Field(default=2.0, ge=0.0, le=20.0, description="boundary：边界像素额外权重")


class TrainConfig(BaseModel):
    """训练参数配置（硅片缺陷检测优化预设）"""
    # 训练模式
    train_mode: str = Field(default="scratch", description="scratch=从头训练, finetune=继承训练, custom=自定义预训练模型")
    resume_from_task_id: Optional[int] = Field(default=None, description="继承训练时，从哪个任务的模型开始")
    resume_model_type: str = Field(default="best", description="继承的模型类型: best/last")

    # 模型选择
    model_name: str = Field(default="yolo26s-seg", description="模型尺寸: yolo26n/s/m/l/x-seg")
    # 自定义预训练模型（train_mode='custom'）：服务器本地 .pt 路径
    model_path: Optional[str] = Field(default=None, description="自定义预训练权重路径（weights/ / 仓库根 / storage/models）")

    # 实例分割掩膜损失；default 保持 Ultralytics 原版行为
    seg_loss: Literal[
        "default", "bce_dice", "focal", "tversky", "focal_tversky", "boundary"
    ] = Field(default="default", description="seg 任务的实例掩膜损失选项")
    seg_loss_params: SegLossParams = Field(
        default_factory=SegLossParams, description="实例掩膜损失超参"
    )

    # 数据划分
    # cls 闭集复核训练允许 1.0：同一批人工复核小图同时作为训练/验证集，
    # 用于验收与人工标签的一致率；其它任务仍沿用常规划分逻辑。
    train_ratio: float = Field(default=0.8, ge=0.5, le=1.0)
    oversample_factor: int = Field(default=5, ge=1, le=20)

    # 基本训练
    epochs: int = Field(default=200, ge=1, le=2000)
    batch_size: int = Field(default=16, ge=1, le=512)   # 大显存机器（≥32G）+ cls 224 可上 256~512
    patience: int = Field(default=50, ge=0)
    device: str = Field(default="auto", description='训练设备：auto=有 CUDA 用 GPU 否则 CPU，也可填 cpu / 0 / cuda:0')
    workers: int = Field(default=4, ge=0, le=16)

    # 学习率
    lr0: float = Field(default=0.01)
    lrf: float = Field(default=0.01)
    momentum: float = Field(default=0.937)
    weight_decay: float = Field(default=0.0005)
    warmup_epochs: float = Field(default=3.0)
    warmup_momentum: float = Field(default=0.8)

    # 颜色增广
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    # 颜色增广模式：hsv=原生 HSV 三强度；rgb=训练时对 R/G/B 分别独立做亮度增益（不改色调/饱和度）；
    # off=关闭颜色增广（hsv_* 与 rgb_* 都不生效）
    color_aug_mode: Literal["hsv", "rgb", "off"] = Field(default="hsv", description="颜色增广模式")
    rgb_gain_p: float = Field(default=0.5, ge=0.0, le=1.0, description="rgb 模式：每个样本生效的概率")
    rgb_gain_max_r: float = Field(default=0.2, ge=0.0, le=1.0, description="rgb 模式：R 通道最大增益幅度（0=该通道不增广）")
    rgb_gain_max_g: float = Field(default=0.2, ge=0.0, le=1.0, description="rgb 模式：G 通道最大增益幅度（0=该通道不增广）")
    rgb_gain_max_b: float = Field(default=0.2, ge=0.0, le=1.0, description="rgb 模式：B 通道最大增益幅度（0=该通道不增广）")

    # 几何增广
    degrees: float = 180.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    flipud: float = 0.5
    fliplr: float = 0.5

    # 高级增广
    mosaic: float = 0.5
    mixup: float = 0.1
    copy_paste: float = 0.5
    erasing: float = 0.1
    close_mosaic: int = 30

    # 形态学预处理
    use_morphology: bool = True
    dilate_kernel: int = Field(default=3, ge=1, le=15)
    erode_kernel: int = Field(default=3, ge=1, le=15)
    mask_dilate_kernel: int = Field(default=0, ge=0, le=15)

    # ---- 语义分割（task_type=semseg，ResUNet34 + clDice）----
    patch_size: int = Field(default=512, ge=128, le=2048, description="semseg: 缺陷中心 patch 尺寸")
    semseg_mode: Literal["binary", "multiclass"] = Field(
        default="binary",
        description="semseg: binary=缺陷/背景二值；multiclass=背景+项目每个类别各成一类",
    )
    semseg_fg_class_weight: float = Field(
        default=20.0, ge=1.0, le=100.0,
        description="semseg 多分类: 前景类的 CE 权重（背景固定 1）",
    )
    semseg_halo: int = Field(
        default=0, ge=0, le=256,
        description="semseg: Halo 上下文像素（输入 = patch_size + 2×halo，0=关闭；建议 8 的倍数）",
    )
    semseg_stride_ratio: float = Field(
        default=0.75, ge=0.25, le=1.0,
        description="semseg: 推理滑窗步长比例（步长 = patch_size × 该值）",
    )
    patches_per_image: int = Field(default=4, ge=1, le=64, description="semseg: 每图每 epoch 的 patch 数")
    accum_steps: int = Field(default=4, ge=1, le=64, description="semseg: 梯度累积步数")
    lr: float = Field(default=1e-4, gt=0, le=0.01, description="semseg: AdamW 初始学习率")
    encoder_lr_mult: float = Field(default=0.1, gt=0, le=1, description="semseg: 编码器学习率倍率")
    pos_weight: float = Field(default=5.0, ge=1.0, le=50.0, description="semseg: BCE 正样本权重")
    cldice_weight: float = Field(default=1.0, ge=0.0, le=10.0, description="semseg: clDice 权重")
    warmup_iters: int = Field(default=500, ge=0, le=100000, description="semseg: 线性预热 iter")
    semseg_threshold: float = Field(default=0.6, ge=0.05, le=0.95, description="semseg: 推理阈值")
    semseg_include_ok: bool = Field(default=False, description="semseg: 无缺陷图是否加入训练")
    semseg_min_area: int = Field(default=0, ge=0, le=100000, description="semseg: 丢弃小于该面积的连通域")
    pretrained: bool = Field(default=True, description="semseg: 编码器是否加载 ImageNet 预训练")


class TrainTaskCreate(BaseModel):
    task_name: str = Field(max_length=200)
    config: TrainConfig = Field(default_factory=TrainConfig)


class TrainTaskOut(BaseModel):
    id: int
    project_id: int
    task_name: str
    status: str
    epochs: int
    current_epoch: int
    best_map50: Optional[float]
    best_fitness: Optional[float] = None
    best_model_path: Optional[str]
    last_model_path: Optional[str]
    output_dir: Optional[str]
    error_message: Optional[str]
    config: Optional[dict] = None
    num_classes: Optional[int] = None
    # 排队位置（1 基）：只有 status='pending' 且在队列里的任务才有值
    queue_position: Optional[int] = None
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    created_at: datetime
    model_config = {"from_attributes": True}


class EpochLogOut(BaseModel):
    epoch: int
    train_box_loss: Optional[float]
    train_seg_loss: Optional[float]
    train_cls_loss: Optional[float]
    train_dfl_loss: Optional[float]
    val_box_loss: Optional[float]
    val_seg_loss: Optional[float]
    val_cls_loss: Optional[float]
    val_dfl_loss: Optional[float]
    precision_b: Optional[float]
    recall_b: Optional[float]
    map50_b: Optional[float]
    map50_95_b: Optional[float]
    map50_m: Optional[float]
    map50_95_m: Optional[float]
    top1_acc: Optional[float] = None   # cls top1
    top5_acc: Optional[float] = None   # cls top5
    val_dice: Optional[float] = None   # semseg Dice
    val_iou: Optional[float] = None    # semseg IoU
    val_recall: Optional[float] = None # semseg pixel recall
    val_miou: Optional[float] = None   # semseg 多分类 mIoU
    val_mdice: Optional[float] = None  # semseg 多分类 mDice
    lr: Optional[float]
    model_config = {"from_attributes": True}
