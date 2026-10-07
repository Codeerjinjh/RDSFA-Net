import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import HGBlock
def _cast_like(x, module):
    """Cast input x to match module parameter dtype (handles AMP/half mismatch in val)."""
    p = next(module.parameters(), None)
    if p is not None and x.dtype != p.dtype:
        return x.to(p.dtype)
    return x




# ========== Helper modules ==========

class SplitRGB(nn.Module):
    """Take first half of channels (for cross-modal interaction output)."""
    def forward(self, x):
        return x[:, :x.shape[1] // 2, :, :].contiguous()


class SplitIR(nn.Module):
    """Take second half of channels (for cross-modal interaction output)."""
    def forward(self, x):
        return x[:, x.shape[1] // 2:, :, :].contiguous()


class SpatialAttention(nn.Module):
    """Spatial attention with 3x3 conv."""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(attn)


class ChannelSEAttention(nn.Module):
    """Squeeze-and-Excitation with 1x1 convs."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class ConvDW(nn.Module):
    """Depthwise separable conv (pointwise + depthwise)."""
    def __init__(self, c1, c2, k=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, stride, padding, groups=c1, bias=bias)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


# ========== ModalityHGBlock (original innovation 1) ==========

class ModalityHGBlock_RGB(HGBlock):
    """RGB branch: HGBlock + SpatialAttention."""
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__(c1, cm, c2, k, n, lightconv, shortcut, act)
        self.spatial_att = SpatialAttention(kernel_size=3)

    def forward(self, x):
        out = super().forward(x)
        return self.spatial_att(out)


class ModalityHGBlock_IR(HGBlock):
    """IR branch: HGBlock + ChannelSEAttention."""
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__(c1, cm, c2, k, n, lightconv, shortcut, act)
        self.channel_att = ChannelSEAttention(c2)

    def forward(self, x):
        out = super().forward(x)
        return self.channel_att(out)


# ========== Early Cross-Modal Interaction (NEW) ==========

class CrossModalInteraction(nn.Module):
    """
    Lightweight cross-modal feature interaction.
    Takes RGB and IR features, exchanges spatial attention information.
    Output: 2 * input_channels (concatenated RGB_out + IR_out).
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Spatial attention generators for each modality
        self.att_rgb = nn.Sequential(
            nn.Conv2d(channels, max(channels // 8, 4), 1, bias=False),
            nn.BatchNorm2d(max(channels // 8, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 4), 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.att_ir = nn.Sequential(
            nn.Conv2d(channels, max(channels // 8, 4), 1, bias=False),
            nn.BatchNorm2d(max(channels // 8, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 4), 1, 1, bias=False),
            nn.Sigmoid(),
        )

        # Channel-wise cross-modal affinity
        self.proj_rgb = nn.Conv2d(channels, max(channels // 16, 4), 1, bias=False)
        self.proj_ir = nn.Conv2d(channels, max(channels // 16, 4), 1, bias=False)

        # Adaptive gating
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, max(channels // 4, 8), 1, bias=False),
            nn.BatchNorm2d(max(channels // 4, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x is a list [f_rgb, f_ir]"""
        f_rgb, f_ir = x[0], x[1]

        # Spatial attention maps (interpretable for paper)
        att_rgb = self.att_rgb(f_rgb)
        att_ir = self.att_ir(f_ir)

        # Cross-modal exchange
        f_rgb_attended = f_rgb * att_ir
        f_ir_attended = f_ir * att_rgb

        # Channel-wise cross-modal affinity
        proj_rgb = self.proj_rgb(f_rgb)
        proj_ir = self.proj_ir(f_ir)
        affinity = torch.sigmoid((proj_rgb * proj_ir).sum(dim=1, keepdim=True))

        # Gate: learn adaptive mixing ratio
        gate_map = self.gate(torch.cat([f_rgb, f_ir], dim=1))
        g_rgb = gate_map[:, 0:1, :, :]
        g_ir = gate_map[:, 1:2, :, :]

        # Residual mixing
        f_rgb_out = f_rgb + f_rgb_attended * g_rgb * affinity
        f_ir_out = f_ir + f_ir_attended * g_ir * affinity

        return torch.cat([f_rgb_out, f_ir_out], dim=1)


# ========== Enhanced Alignment: Multi-scale Deformable Alignment ==========

class MultiScaleDeformableAlignment(nn.Module):
    """
    Multi-scale deformable alignment with dilated convs.
    3 branches: fine (3x3), medium (d=2), coarse (d=3).
    Group convs for efficiency, wider channels for capacity.
    """
    def __init__(self, channels):
        super().__init__()
        self.scale = 0.15
        mid = max(channels // 2, 16)

        self.shared = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1, groups=4, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # Multi-scale offset branches
        self.offset_b1 = nn.Sequential(  # fine: 3x3
            nn.Conv2d(mid, mid, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 3, padding=1, bias=False),
        )
        self.offset_b2 = nn.Sequential(  # medium: d=2
            nn.Conv2d(mid, mid, 3, padding=2, dilation=2, groups=4, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 3, padding=2, dilation=2, bias=False),
        )
        self.offset_b3 = nn.Sequential(  # coarse: d=3
            nn.Conv2d(mid, mid, 3, padding=3, dilation=3, groups=4, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 3, padding=3, dilation=3, bias=False),
        )

        # Adaptive fusion weights
        self.fusion_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, max(mid // 4, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(mid // 4, 4), 3, 1, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, f_rgb, f_ir):
        # Handle dtype mismatch (validation may cast model to half)
        B, _, H, W = f_rgb.shape
        f_rgb = _cast_like(f_rgb, self)
        f_ir = _cast_like(f_ir, self)
        # Conv ops run in original dtype (fp16 under AMP)
        shared = self.shared(torch.cat([f_rgb, f_ir], dim=1))

        offset1 = torch.tanh(self.offset_b1(shared))
        offset2 = torch.tanh(self.offset_b2(shared))
        offset3 = torch.tanh(self.offset_b3(shared))

        weights = self.fusion_weight(shared)
        offset = offset1 * weights[:, 0:1] + offset2 * weights[:, 1:2] + offset3 * weights[:, 2:3]

        # Grid_sample only: cast to fp32 for numerical stability under AMP
        is_half = offset.dtype == torch.float16
        if is_half:
            offset_f = offset.float()
            f_rgb_f = f_rgb.float()
            f_ir_f = f_ir.float()
            grid_dtype = torch.float32
        else:
            offset_f = offset
            f_rgb_f = f_rgb
            f_ir_f = f_ir
            grid_dtype = f_rgb.dtype

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=f_rgb.device, dtype=grid_dtype),
            torch.linspace(-1, 1, W, device=f_rgb.device, dtype=grid_dtype),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        offset_perm = offset_f.permute(0, 2, 3, 1) * self.scale

        f_rgb_aligned = F.grid_sample(
            f_rgb_f, grid + offset_perm, mode="bilinear", padding_mode="border", align_corners=True
        )
        f_ir_aligned = F.grid_sample(
            f_ir_f, grid - offset_perm, mode="bilinear", padding_mode="border", align_corners=True
        )

        if is_half:
            f_rgb_aligned = f_rgb_aligned.half()
            f_ir_aligned = f_ir_aligned.half()
        return f_rgb_aligned, f_ir_aligned


# ========== Enhanced Degradation Estimator ==========

class EnhancedDegradationEstimator(nn.Module):
    """Wider intermediate channels (ch//3 vs original ch//4)."""
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 3, 12)
        self.estimator = nn.Sequential(
            ConvDW(channels * 2, mid, 3, 1, 1),
            ConvDW(mid, mid, 3, 1, 1),
            nn.Conv2d(mid, 5, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f_rgb, f_ir):
        f_rgb = _cast_like(f_rgb, self)
        f_ir = _cast_like(f_ir, self)
        cat = torch.cat([f_rgb, f_ir], dim=1)
        out = self.estimator(cat)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], out[:, 4:5]


# ========== Dynamic Modality Gate (NEW) ==========

class DynamicModalityGate(nn.Module):
    """
    Per-location modality confidence weighting.
    Output: [0,1] weight maps for RGB and IR.
    Interpretable: can visualize which modality is trusted where.
    """
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 8, 4)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 3, padding=1, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, f_rgb, f_ir):
        f_rgb = _cast_like(f_rgb, self)
        f_ir = _cast_like(f_ir, self)
        gate_map = self.gate(torch.cat([f_rgb, f_ir], dim=1))
        w_rgb = gate_map[:, 0:1, :, :]
        w_ir = gate_map[:, 1:2, :, :]
        f_fused = f_rgb * w_rgb + f_ir * w_ir
        return f_fused, w_rgb, w_ir


# ========== Enhanced Recovery-Guided Fusion ==========

class EnhancedRecoveryGuidedFusion(nn.Module):
    """
    Enhanced fusion: multi-branch aggregation + dynamic gating.
    Outputs C channels.
    """
    def __init__(self, channels):
        super().__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.priv_rgb_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )
        self.priv_ir_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )

        # Multi-branch aggregation: 1x1 + 3x3 depthwise
        self.agg_1x1 = nn.Conv2d(channels * 2, channels // 2, 1, bias=False)
        self.agg_3x3_dw = nn.Conv2d(
            channels * 2, channels // 2, 3, padding=1,
            groups=max((channels * 2) // 8, 1), bias=False
        )

        # Dynamic gating
        self.dynamic_gate = DynamicModalityGate(channels)

        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, f_rgb, f_ir, D_rgb, D_ir, D_blur, W_rgb_rec, W_ir_rec):
        f_rgb = _cast_like(f_rgb, self)
        f_ir = _cast_like(f_ir, self)
        shared_mix = (
            f_rgb * (1 - W_rgb_rec)
            + f_ir * W_rgb_rec
            + f_ir * (1 - W_ir_rec)
            + f_rgb * W_ir_rec
        )
        shared_mix = shared_mix * (1 - D_blur) + 0.5 * (f_rgb + f_ir) * D_blur
        Fs = self.shared_conv(shared_mix)
        Fr = self.priv_rgb_conv(f_rgb * (1 - D_rgb))
        Fi = self.priv_ir_conv(f_ir * (1 - D_ir))

        # Multi-branch aggregation
        cat_sp = torch.cat([Fs, torch.cat([Fr, Fi], dim=1)], dim=1)
        agg_1 = self.agg_1x1(cat_sp)
        agg_3 = self.agg_3x3_dw(cat_sp)
        agg = torch.cat([agg_1, agg_3], dim=1)

        # Dynamic gating
        f_gated, w_rgb, w_ir = self.dynamic_gate(f_rgb, f_ir)

        Ffusion = self.fusion_proj(torch.cat([agg, f_gated], dim=1))
        return Ffusion, w_rgb, w_ir


# ========== Cross-Scale Refinement (for cross-scale path) ==========

class CrossScaleFusion(nn.Module):
    """
    Refines current-level fusion output using previous-level (higher-res) info.
    """
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels_out * 2, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(channels_out),
        )

    def forward(self, x):
        """x is a list [feat_prev, feat_curr]"""
        feat_prev, feat_curr = x
        feat_prev = _cast_like(feat_prev, self)
        feat_curr = _cast_like(feat_curr, self)
        if feat_prev.shape[-2:] != feat_curr.shape[-2:]:
            feat_prev = F.interpolate(feat_prev, size=feat_curr.shape[-2:], mode="bilinear", align_corners=False)
        feat_prev = self.proj(feat_prev)
        return self.refine(torch.cat([feat_prev, feat_curr], dim=1)) + feat_curr


# ========== Main: Enhanced RDRFusion (v2) ==========

class RDRFusion(nn.Module):
    """
    Enhanced RDRFA Fusion v2.
    Accepts 2 inputs [f_rgb, f_ir].
    Improvements: multi-scale alignment, wider degradation estimator,
                  dynamic modality gating, multi-branch aggregation.
    Cross-scale refinement is handled externally by CrossScaleFusion.
    """
    def __init__(self, Channel):
        super().__init__()
        self.alignment = MultiScaleDeformableAlignment(Channel)
        self.degradation = EnhancedDegradationEstimator(Channel)
        self.fusion = EnhancedRecoveryGuidedFusion(Channel)

        # Cross-scale modulation: global-context gate from previous level features
        self.cross_mod = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(Channel, Channel, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        """inputs: [f_rgb, f_ir] - fuses modalities at one scale."""
        f_rgb, f_ir = inputs[0], inputs[1]
        f_rgb = _cast_like(f_rgb, self)
        f_ir = _cast_like(f_ir, self)

        # Multi-scale alignment
        f_rgb_aligned, f_ir_aligned = self.alignment(f_rgb, f_ir)

        # Degradation estimation
        D_rgb, D_ir, D_blur, R_rgb, R_ir = self.degradation(f_rgb_aligned, f_ir_aligned)

        # Recovery
        f_rgb_clean = f_rgb_aligned * R_rgb
        f_ir_clean = f_ir_aligned * R_ir

        # Enhanced fusion
        Ffusion, w_rgb, w_ir = self.fusion(
            f_rgb_clean, f_ir_clean, D_rgb, D_ir, D_blur,
            1 - D_rgb, 1 - D_ir
        )

        return Ffusion

class RDSFusion(RDRFusion):
    pass
