# -*- coding: utf-8 -*-
"""ResUNet34 + FCNHead 语义分割网络（MMSegmentation 结构的逐模块复刻）。

结构来源：``scratch_seg/scratchseg/models/resunet.py`` +
MMSegmentation v1.2.2 的 ``FCNHead``。

**模块命名与 MMSeg checkpoint 完全一致**，因此 ``state_dict`` 可以直接
按同名键加载，不需要任何重命名映射：

    backbone.enc0.0.weight            torchvision resnet34 conv1
    backbone.enc0.1.*                 resnet34 bn1
    backbone.enc1.1.*                 resnet34 layer1（enc1.0 是 maxpool，无参数）
    backbone.enc2.* / enc3.* / enc4.* layer2 / layer3 / layer4
    backbone.up4.conv.{0,1,3,4}.*     ConvBlock：conv-GN-ReLU-conv-GN-ReLU
    decode_head.convs.{0,1}.conv.*    FCNHead 的 2 个 3x3 ConvModule
    decode_head.conv_cat.conv.*       cat(输入, 卷积输出) 后的 3x3 ConvModule
    decode_head.conv_seg.*            1x1 分类卷积

重要口径（勿改）：
1. 网络输出是 **stride-2** 的 logits（512 输入 → 256 输出），最终上采样在
   推理侧做（与实验 ``evaluate.py`` / ``export_onnx.py`` 一致）。
2. ``norm_eval=True`` 时训练也把 BatchNorm 固定为 eval（实验同样覆写了
   ``train()``），这样 batch=1 时 BN 统计量不会漂移。
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Conv2d + (可选 Norm) + ReLU，键名与 mmcv 的 ConvModule 兼容（``.conv``）。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 0, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, bias=bias,
        )
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.conv(x))


class ConvBlock(nn.Module):
    """3x3 conv + GroupNorm(8) + ReLU 堆叠两次（实验里的 U 型解码单元）。

    命名必须是 ``self.conv = Sequential(...)``，索引 0/1/3/4 分别是
    conv/gn/conv/gn，才能对上 ``up4.conv.0.weight`` 这类键名。
    """

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResUNet34Backbone(nn.Module):
    """torchvision ResNet34 编码器 + U 型跳跃解码，输出 stride-2 的 64 通道特征。"""

    def __init__(self, pretrained: bool = False, norm_eval: bool = True):
        super().__init__()
        from torchvision.models import resnet34

        weights = None
        if pretrained:
            try:
                from torchvision.models import ResNet34_Weights
                weights = ResNet34_Weights.IMAGENET1K_V1
            except Exception as exc:      # 离线环境拿不到 torchvision 枚举
                print(f"[semseg] 预训练权重不可用，改用随机初始化：{exc}")
        rn = resnet34(weights=weights)

        self.enc0 = nn.Sequential(rn.conv1, rn.bn1, rn.relu)   # /2
        self.enc1 = nn.Sequential(rn.maxpool, rn.layer1)       # /4
        self.enc2 = rn.layer2                                  # /8
        self.enc3 = rn.layer3                                  # /16
        self.enc4 = rn.layer4                                  # /32

        c = [64, 64, 128, 256, 512]
        self.up4 = ConvBlock(c[4] + c[3], 256)   # /16
        self.up3 = ConvBlock(256 + c[2], 128)    # /8
        self.up2 = ConvBlock(128 + c[1], 64)     # /4
        self.up1 = ConvBlock(64 + c[0], 64)      # /2
        self.norm_eval = norm_eval

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        d = self.up4(torch.cat([self._up(x4, x3), x3], dim=1))
        d = self.up3(torch.cat([self._up(d, x2), x2], dim=1))
        d = self.up2(torch.cat([self._up(d, x1), x1], dim=1))
        d = self.up1(torch.cat([self._up(d, x0), x0], dim=1))
        return (d,)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self


class FCNHead(nn.Module):
    """MMSeg v1.2.2 FCNHead：num_convs=2、concat_input=True、无 dropout、无上采样。"""

    def __init__(self, in_channels: int = 64, channels: int = 64,
                 num_classes: int = 1, num_convs: int = 2,
                 concat_input: bool = True, in_index: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.concat_input = concat_input
        self.in_index = in_index

        convs = [
            ConvModule(
                in_channels if i == 0 else channels,
                channels,
                kernel_size=3,
                padding=1,
            )
            for i in range(num_convs)
        ]
        self.convs = nn.Sequential(*convs)
        if concat_input:
            self.conv_cat = ConvModule(in_channels + channels, channels,
                                       kernel_size=3, padding=1)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def _transform_inputs(self, inputs) -> torch.Tensor:
        if isinstance(inputs, (list, tuple)):
            return inputs[self.in_index]
        return inputs

    def forward(self, inputs) -> torch.Tensor:
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return self.conv_seg(feats)


class ResUNet34Segmenter(nn.Module):
    """完整分割网络。``forward`` 返回 stride-2 logits（未 sigmoid、未上采样）。

    ``num_classes > 1`` 时是**多分类**模式（0=背景，1..N=缺陷类，与 ``masks_ids``
    的像素值一致），概率图用 softmax；``num_classes == 1`` 时是原来的二值模式，
    概率图用 sigmoid。
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = False,
                 norm_eval: bool = True):
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone = ResUNet34Backbone(pretrained=pretrained, norm_eval=norm_eval)
        self.decode_head = FCNHead(
            in_channels=64,
            channels=64,
            num_classes=num_classes,
            num_convs=2,
            concat_input=True,
            in_index=0,
        )

    @property
    def is_multiclass(self) -> bool:
        return self.num_classes > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_head(self.backbone(x))

    def forward_logits_full(self, x: torch.Tensor) -> torch.Tensor:
        """上采样回输入分辨率的 logits（导出/部署用）。"""
        logits = self.forward(x)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward_prob(self, x: torch.Tensor) -> torch.Tensor:
        """概率图，等价于实验的 ExportWrapper（上采样 + sigmoid/softmax）。

        多分类输出 ``(B,N,H,W)`` 各类概率（沿 dim=1 和为 1）；
        二分类输出 ``(B,1,H,W)`` 的前景概率。
        """
        logits = self.forward_logits_full(x)
        if self.is_multiclass:
            return torch.softmax(logits, dim=1)
        return torch.sigmoid(logits)


def build_model(num_classes: int = 1, pretrained: bool = False,
                norm_eval: bool = True) -> ResUNet34Segmenter:
    """构造模型（默认不下载 ImageNet 权重，避免离线机器卡住）。

    ``num_classes=1`` → 二值缺陷分割；``num_classes=N+1`` → 背景 + N 类缺陷。
    """
    return ResUNet34Segmenter(
        num_classes=num_classes,
        pretrained=pretrained,
        norm_eval=norm_eval,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
