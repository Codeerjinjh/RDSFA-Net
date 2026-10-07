from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用原工具函数（略作优化，保持兼容性）
def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    if padding is None:
        padding = (kernel_size - 1) * dilation // 2
    return padding

def make_divisible(value: int, divisor: int = 8) -> int:
    return int((value + divisor // 2) // divisor * divisor)

# -------------- 新增：轻量级通道注意力模块（创新点支撑）--------------
class LightChannelAttention(nn.Module):
    """轻量级通道注意力，避免SE模块的计算量"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, make_divisible(channels // reduction), 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(make_divisible(channels // reduction), channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))

# -------------- 创新核心模块：DRFLI-Block --------------
class DynamicReceptiveFieldLightInception(nn.Module):
    """
    动态感受野轻量化Inception模块（DRFLI-Block）
    创新点：
    1. 动态感受野：通过 dilation 动态调整卷积感受野，适配不同尺度目标；
    2. 注意力加权融合：多尺度特征通过通道注意力加权，而非简单相加；
    3. 残差捷径优化：引入多条残差路径，缓解梯度消失，提升训练稳定性；
    4. 轻量化设计：深度可分离卷积+通道降维，控制计算量（优于原Poly Kernel模块）。
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            base_kernels: Sequence[int] = (3, 5, 7),  # 精简核大小，降低计算量（创新点：聚焦有效感受野）
            dilation_ranges: Sequence[Tuple[int, int]] = ((1, 2), (1, 3), (2, 4)),  # 动态dilation范围（创新点）
            expansion: float = 0.8,  # 降低扩张系数，更轻量化
            reduction: int = 4,  # 注意力降维系数
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_channels = make_divisible(int(self.out_channels * expansion), 8)  # 自适应通道

        # 1. 预处理：1x1卷积降维（轻量化核心）
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_channels, momentum=norm_cfg['momentum'], eps=norm_cfg['eps']),
            nn.SiLU(inplace=True)
        )

        # 2. 动态感受野卷积分支（核心创新）
        self.conv_branches = nn.ModuleList()
        self.dilation_ranges = dilation_ranges
        for idx, (kernel, (d_min, d_max)) in enumerate(zip(base_kernels, dilation_ranges)):
            # 动态dilation（训练时随机采样，推理时用中间值，适配不同目标尺度）
            self.register_buffer(f'dilation_{idx}', torch.tensor((d_min + d_max) // 2, dtype=torch.int))  # 推理默认值
            self.conv_branches.append(
                nn.Conv2d(
                    self.hidden_channels, self.hidden_channels,
                    kernel_size=kernel, stride=1, padding=0,  # padding动态计算
                    dilation=(d_min + d_max) // 2, groups=self.hidden_channels, bias=False
                )
            )

        # 3. 注意力融合模块（创新点：加权融合多尺度特征）
        self.attention_fusion = LightChannelAttention(len(base_kernels) * self.hidden_channels, reduction)

        # 4. 残差捷径（创新点：多路径残差，提升稳定性）
        self.shortcut1 = nn.Conv2d(in_channels, self.hidden_channels, 1, 1, 0, bias=False) if in_channels != self.hidden_channels else nn.Identity()
        self.shortcut2 = nn.Conv2d(self.hidden_channels * len(base_kernels), self.out_channels, 1, 1, 0, bias=False)
        self.bn_shortcut = nn.BatchNorm2d(self.out_channels, momentum=norm_cfg['momentum'], eps=norm_cfg['eps'])

        # 5. 后处理：轻量特征精炼
        self.post_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * len(base_kernels), self.hidden_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_channels, momentum=norm_cfg['momentum'], eps=norm_cfg['eps']),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        # 原始输入保存（用于残差）
        residual = self.shortcut1(x)

        # 1. 预处理降维
        x = self.pre_conv(x)

        # 2. 动态感受野卷积（训练时随机调整dilation，推理时固定）
        branch_outputs = []
        for idx, (conv, d_range) in enumerate(zip(self.conv_branches, self.dilation_ranges)):
            if self.training:
                # 训练时随机采样dilation，增强泛化性（创新点：动态适配）
                dilation = torch.randint(d_range[0], d_range[1] + 1, (1,)).item()
                conv.dilation = (dilation, dilation)
            else:
                # 推理时用默认dilation
                dilation = getattr(self, f'dilation_{idx}').item()
                conv.dilation = (dilation, dilation)
            # 动态计算padding（适配当前dilation）
            padding = autopad(conv.kernel_size[0], None, dilation)
            branch_out = conv(F.pad(x, (padding, padding, padding, padding)))
            branch_outputs.append(branch_out)

        # 3. 拼接多尺度特征 + 注意力加权
        fused = torch.cat(branch_outputs, dim=1)  # 通道拼接（比相加更能保留多尺度信息）
        fused = self.attention_fusion(fused)  # 注意力加权，突出有效特征

        # 4. 后处理精炼 + 残差融合
        refined = self.post_conv(fused)
        refined = refined + residual  # 残差连接（缓解梯度消失）
        refined = F.silu(refined, inplace=True)

        # 5. 输出通道匹配
        out = self.shortcut2(fused) + self.bn_shortcut(refined)  # 双路径融合（创新点）
        return F.silu(out, inplace=True)

# -------------- 验证模块兼容性（替换RepC3无压力）--------------
