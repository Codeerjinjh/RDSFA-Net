# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad, DSConv
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "IR_HGBlock",
    "RGB_HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "HyperACE", 
    "DownsampleConv", 
    "FullPAD_Tunnel",
    "DSC3k2"
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y



class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


# class RepC3(nn.Module):
#     """Rep C3."""

#     def __init__(self, c1, c2, n=3, e=1.0):
#         """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c2, 1, 1)
#         self.cv2 = Conv(c1, c2, 1, 1)
#         self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
#         self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

#     def forward(self, x):
#         """Forward pass of RT-DETR neck layer."""
#         return self.cv3(self.m(self.cv1(x)) + self.cv2(x))
class RepC3(nn.Module):
    """Rep C3."""
 
    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()
 
    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))



class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class DSBottleneck(nn.Module):
    """
    An improved bottleneck block using depthwise separable convolutions (DSConv).

    This class implements a lightweight bottleneck module that replaces standard convolutions with depthwise
    separable convolutions to reduce parameters and computational cost. 

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to use a residual shortcut connection. The connection is only added if c1 == c2. Defaults to True.
        e (float, optional): Expansion ratio for the intermediate channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv layer. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv layer. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv layer. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSBottleneck module.

    Examples:
        >>> import torch
        >>> model = DSBottleneck(c1=64, c2=64, shortcut=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)   
        self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)  
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k(C3):
    """
    An improved C3k module using DSBottleneck blocks for lightweight feature extraction.

    This class extends the C3 module by replacing its standard bottleneck blocks with DSBottleneck blocks,
    which use depthwise separable convolutions.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of DSBottleneck blocks to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections within the DSBottlenecks. Defaults to True.
        g (int, optional): Number of groups for grouped convolution (passed to parent C3). Defaults to 1.
        e (float, optional): Expansion ratio for the C3 module's hidden channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv in each DSBottleneck. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in each DSBottleneck. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv in each DSBottleneck. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k module (inherited from C3).

    Examples:
        >>> import torch
        >>> model = DSC3k(c1=128, c2=128, n=2, k1=3, k2=7)
        >>> x = torch.randn(2, 128, 64, 64)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 64, 64])
    """
    def __init__(
        self,
        c1,                
        c2,                 
        n=1,                
        shortcut=True,      
        g=1,                 
        e=0.5,              
        k1=3,               
        k2=5,               
        d2=1                 
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  

        self.m = nn.Sequential(
            *(
                DSBottleneck(
                    c_, c_,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        )

class DSC3k2(C2f):
    """
    An improved C3k2 module that uses lightweight depthwise separable convolution blocks.

    This class redesigns C3k2 module, replacing its internal processing blocks with either DSBottleneck
    or DSC3k modules.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of internal processing blocks to stack. Defaults to 1.
        dsc3k (bool, optional): If True, use DSC3k as the internal block. If False, use DSBottleneck. Defaults to False.
        e (float, optional): Expansion ratio for the C2f module's hidden channels. Defaults to 0.5.
        g (int, optional): Number of groups for grouped convolution (passed to parent C2f). Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections in the internal blocks. Defaults to True.
        k1 (int, optional): Kernel size for the first DSConv in internal blocks. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in internal blocks. Defaults to 7.
        d2 (int, optional): Dilation for the second DSConv in internal blocks. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k2 module (inherited from C2f).

    Examples:
        >>> import torch
        >>> # Using DSBottleneck as internal block
        >>> model1 = DSC3k2(c1=64, c2=64, n=2, dsc3k=False)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output1 = model1(x)
        >>> print(f"With DSBottleneck: {output1.shape}")
        With DSBottleneck: torch.Size([2, 64, 128, 128])
        >>> # Using DSC3k as internal block
        >>> model2 = DSC3k2(c1=64, c2=64, n=1, dsc3k=True)
        >>> output2 = model2(x)
        >>> print(f"With DSC3k: {output2.shape}")
        With DSC3k: torch.Size([2, 64, 128, 128])
    """
    def __init__(
        self,
        c1,          
        c2,         
        n=1,          
        dsc3k=False,  
        e=0.5,       
        g=1,        
        shortcut=True,
        k1=3,       
        k2=7,       
        d2=1         
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k(
                    self.c, self.c,
                    n=2,           
                    shortcut=shortcut,
                    g=g,
                    e=1.0,  
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                DSBottleneck(
                    self.c, self.c,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )

class AdaHyperedgeGen(nn.Module):
    """
    Generates an adaptive hyperedge participation matrix from a set of vertex features.

    This module implements the Adaptive Hyperedge Generation mechanism. It generates dynamic hyperedge prototypes
    based on the global context of the input nodes and calculates a continuous participation matrix (A)
    that defines the relationship between each vertex and each hyperedge.

    Attributes:
        node_dim (int): The feature dimension of each input node.
        num_hyperedges (int): The number of hyperedges to generate.
        num_heads (int, optional): The number of attention heads for multi-head similarity calculation. Defaults to 4.
        dropout (float, optional): The dropout rate applied to the logits. Defaults to 0.1.
        context (str, optional): The type of global context to use ('mean', 'max', or 'both'). Defaults to "both".

    Methods:
        forward: Takes a batch of vertex features and returns the participation matrix A.

    Examples:
        >>> import torch
        >>> model = AdaHyperedgeGen(node_dim=64, num_hyperedges=16, num_heads=4)
        >>> x = torch.randn(2, 100, 64)  # (Batch, Num_Nodes, Node_Dim)
        >>> A = model(x)
        >>> print(A.shape)
        torch.Size([2, 100, 16])
    """
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)  
        elif context == "both":
            self.context_net = nn.Linear(2*node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(
                f"Unsupported context '{context}'. "
                "Expected one of: 'mean', 'max', 'both'."
            )

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
    
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)          
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)          
        else:
            avg_context = X.mean(dim=1)           
            max_context, _ = X.max(dim=1)           
            context_cat = torch.cat([avg_context, max_context], dim=-1) 
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)  
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets           
        
        X_proj = self.pre_head_proj(X) 
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling 
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1) 
        
        logits = self.dropout(logits)  

        return F.softmax(logits, dim=1)

class AdaHGConv(nn.Module):
    """
    Performs the adaptive hypergraph convolution.

    This module contains the two-stage message passing process of hypergraph convolution:
    1. Generates an adaptive participation matrix using AdaHyperedgeGen.
    2. Aggregates vertex features into hyperedge features (vertex-to-edge).
    3. Disseminates hyperedge features back to update vertex features (edge-to-vertex).
    A residual connection is added to the final output.

    Attributes:
        embed_dim (int): The feature dimension of the vertices.
        num_hyperedges (int, optional): The number of hyperedges for the internal generator. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the internal generator. Defaults to 4.
        dropout (float, optional): The dropout rate for the internal generator. Defaults to 0.1.
        context (str, optional): The context type for the internal generator. Defaults to "both".

    Methods:
        forward: Performs the adaptive hypergraph convolution on a batch of vertex features.

    Examples:
        >>> import torch
        >>> model = AdaHGConv(embed_dim=128, num_hyperedges=16, num_heads=8)
        >>> x = torch.randn(2, 256, 128) # (Batch, Num_Nodes, Dim)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 256, 128])
    """
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        
    def forward(self, X):
        A = self.edge_generator(X)  
        
        He = torch.bmm(A.transpose(1, 2), X) 
        He = self.edge_proj(He)
        
        X_new = torch.bmm(A, He)  
        X_new = self.node_proj(X_new)
        
        return X_new + X
        
class AdaHGComputation(nn.Module):
    """
    A wrapper module for applying adaptive hypergraph convolution to 4D feature maps.

    This class makes the hypergraph convolution compatible with standard CNN architectures. It flattens a
    4D input tensor (B, C, H, W) into a sequence of vertices (tokens), applies the AdaHGConv layer to
    model high-order correlations, and then reshapes the output back into a 4D tensor.

    Attributes:
        embed_dim (int): The feature dimension of the vertices (equivalent to input channels C).
        num_hyperedges (int, optional): The number of hyperedges for the underlying AdaHGConv. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the underlying AdaHGConv. Defaults to 8.
        dropout (float, optional): The dropout rate for the underlying AdaHGConv. Defaults to 0.1.
        context (str, optional): The context type for the underlying AdaHGConv. Defaults to "both".

    Methods:
        forward: Processes a 4D feature map through the adaptive hypergraph computation layer.

    Examples:
        >>> import torch
        >>> model = AdaHGComputation(embed_dim=64, num_hyperedges=8, num_heads=4)
        >>> x = torch.randn(2, 64, 32, 32) # (B, C, H, W)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(
            embed_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            dropout=dropout,
            context=context
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) 
        tokens = self.hgnn(tokens) 
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out 

class C3AH(nn.Module):
    """
    A CSP-style block integrating Adaptive Hypergraph Computation (C3AH).

    The input feature map is split into two paths.
    One path is processed by the AdaHGComputation module to model high-order correlations, while the other
    serves as a shortcut. The outputs are then concatenated to fuse features.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float, optional): Expansion ratio for the hidden channels. Defaults to 1.0.
        num_hyperedges (int, optional): The number of hyperedges for the internal AdaHGComputation. Defaults to 8.
        context (str, optional): The context type for the internal AdaHGComputation. Defaults to "both".

    Methods:
        forward: Performs a forward pass through the C3AH module.

    Examples:
        >>> import torch
        >>> model = C3AH(c1=64, c2=128, num_hyperedges=8)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 32, 32])
    """
    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)  
        assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(embed_dim=c_, 
                          num_hyperedges=num_hyperedges, 
                          num_heads=num_heads,
                          dropout=0.1,
                          context=context)
        self.cv3 = Conv(2 * c_, c2, 1)  
        
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class FuseModule(nn.Module):
    """
    A module to fuse multi-scale features for the HyperACE block.

    This module takes a list of three feature maps from different scales, aligns them to a common
    spatial resolution by downsampling the first and upsampling the third, and then concatenates
    and fuses them with a convolution layer.

    Attributes:
        c_in (int): The number of channels of the input feature maps.
        channel_adjust (bool): Whether to adjust the channel count of the concatenated features.

    Methods:
        forward: Fuses a list of three multi-scale feature maps.

    Examples:
        >>> import torch
        >>> model = FuseModule(c_in=64, channel_adjust=False)
        >>> # Input is a list of features from different backbone stages
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, c_in, channel_adjust):
        super(FuseModule, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if channel_adjust:
            self.conv_out = Conv(4 * c_in, c_in, 1)
        else:
            self.conv_out = Conv(3 * c_in, c_in, 1)

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        out = self.conv_out(x_cat)
        return out

class HyperACE(nn.Module):
    """
    Hypergraph-based Adaptive Correlation Enhancement (HyperACE).

    This is the core module of YOLOv13, designed to model both global high-order correlations and
    local low-order correlations. It first fuses multi-scale features, then processes them through parallel
    branches: two C3AH branches for high-order modeling and a lightweight DSConv-based branch for
    low-order feature extraction.

    Attributes:
        c1 (int): Number of input channels for the fuse module.
        c2 (int): Number of output channels for the entire block.
        n (int, optional): Number of blocks in the low-order branch. Defaults to 1.
        num_hyperedges (int, optional): Number of hyperedges for the C3AH branches. Defaults to 8.
        dsc3k (bool, optional): If True, use DSC3k in the low-order branch; otherwise, use DSBottleneck. Defaults to True.
        shortcut (bool, optional): Whether to use shortcuts in the low-order branch. Defaults to False.
        e1 (float, optional): Expansion ratio for the main hidden channels. Defaults to 0.5.
        e2 (float, optional): Expansion ratio within the C3AH branches. Defaults to 1.
        context (str, optional): Context type for C3AH branches. Defaults to "both".
        channel_adjust (bool, optional): Passed to FuseModule for channel configuration. Defaults to True.

    Methods:
        forward: Performs a forward pass through the HyperACE module.

    Examples:
        >>> import torch
        >>> model = HyperACE(c1=64, c2=256, n=1, num_hyperedges=8)
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=True):
        super().__init__()
        self.c = int(c2 * e1) 
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1) 
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)
                    
    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))

class DownsampleConv(nn.Module):
    """
    A simple downsampling block with optional channel adjustment.

    This module uses average pooling to reduce the spatial dimensions (H, W) by a factor of 2. It can
    optionally include a 1x1 convolution to adjust the number of channels, typically doubling them.

    Attributes:
        in_channels (int): The number of input channels.
        channel_adjust (bool, optional): If True, a 1x1 convolution doubles the channel dimension. Defaults to True.

    Methods:
        forward: Performs the downsampling and optional channel adjustment.

    Examples:
        >>> import torch
        >>> model = DownsampleConv(in_channels=64, channel_adjust=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 16, 16])
    """
    def __init__(self, in_channels, channel_adjust=True):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        if channel_adjust:
            self.channel_adjust = Conv(in_channels, in_channels * 2, 1)
        else:
            self.channel_adjust = nn.Identity() 

    def forward(self, x):
        return self.channel_adjust(self.downsample(x))

class FullPAD_Tunnel(nn.Module):
    """
    A gated fusion module for the Full-Pipeline Aggregation-and-Distribution (FullPAD) paradigm.

    This module implements a gated residual connection used to fuse features. It takes two inputs: the original
    feature map and a correlation-enhanced feature map. It then computes `output = original + gate * enhanced`,
    where `gate` is a learnable scalar parameter that adaptively balances the contribution of the enhanced features.

    Methods:
        forward: Performs the gated fusion of two input feature maps.

    Examples:
        >>> import torch
        >>> model = FullPAD_Tunnel()
        >>> original_feature = torch.randn(2, 64, 32, 32)
        >>> enhanced_feature = torch.randn(2, 64, 32, 32)
        >>> output = model([original_feature, enhanced_feature])
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        out = x[0] + self.gate * x[1]
        return out

#####################


def _radial_masks_ir(h, w, lf_cut=0.2, bp_cut=0.4, device=None, dtype=None, smooth=True):
    """
    为 IR 生成三个 mask（低通 / 带通 / 高频）
    lf_cut: 低频截止比例（0..1）
    bp_cut: 带通上界比例（0..1），带通下界为 lf_cut
    """
    ys = torch.arange(h, device=device, dtype=dtype).unsqueeze(1)
    xs = torch.arange(w, device=device, dtype=dtype).unsqueeze(0)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    dy = ys - cy
    dx = xs - cx
    dist = torch.sqrt(dy * dy + dx * dx)
    maxr = torch.sqrt((cy) ** 2 + (cx) ** 2)
    d = dist / (maxr + 1e-8)
    # smooth transitions
    if smooth:
        sigma = 0.02
        mask_lf = torch.sigmoid((lf_cut - d) / sigma)  # 高在中心
        mask_bp = torch.sigmoid((d - lf_cut) / sigma) * torch.sigmoid((bp_cut - d) / sigma)  # 环形
        mask_hf = 1.0 - (mask_lf + mask_bp)
        mask_hf = mask_hf.clamp(min=0.0)
    else:
        mask_lf = (d <= lf_cut).to(dtype=dtype)
        mask_bp = ((d > lf_cut) & (d <= bp_cut)).to(dtype=dtype)
        mask_hf = (d > bp_cut).to(dtype=dtype)
    return mask_lf.unsqueeze(0).unsqueeze(0), mask_bp.unsqueeze(0).unsqueeze(0), mask_hf.unsqueeze(0).unsqueeze(0)


class IRJointFilterModule(nn.Module):
    """
    IR 低通 - 带通 联合滤波模块

    输入 / 输出: (B, C, H, W) -> (B, C, H, W)

    核心流程（在频域执行）：
      1. FFT -> F
      2. 构造低频 / 带通 / 高频 mask
      3. 低频：幅度增强 + 平滑（温度主体增强）
      4. 带通：基于局部相位一致性判断信噪，再选择性保留/增强
      5. 高频：按照距离逐步抑制并可截断极高频（去噪）
      6. 重建 IFFT -> small conv refine -> 返回
    傅里叶变换的作用在于把红外的“温度分布（低频）”与“高频噪声”明确分离，使得能够针对性增强主体并抑制噪声，而不是在空域难以区分。
    """
    def __init__(self, channels, lf_cut=0.2, bp_cut=0.4, hf_trunc=0.8):
        super().__init__()
        self.channels = channels
        self.lf_cut = lf_cut
        self.bp_cut = bp_cut
        self.hf_trunc = hf_trunc

        # 低频增强系数（可学习标量或 per-channel）
        self.lf_gain = nn.Parameter(torch.tensor(1.2), requires_grad=True)  # 初始 1.2 倍
        # 中频增强系数上限
        self.bp_gain = nn.Parameter(torch.tensor(0.6), requires_grad=True)

        # 中频相位一致性判别的平滑阈值（learnable）
        self.phase_thresh = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.phase_smooth = 0.1

        # 高频抑制强度参数（learnable）
        self.hf_gamma = nn.Parameter(torch.tensor(6.0), requires_grad=True)

        # 重建后 refine
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # ---------- 1) FFT ----------
        F = torch.fft.fft2(x, norm='ortho')  # complex (B,C,H,W)
        A = torch.abs(F)
        P = torch.angle(F)

        # ---------- 2) 构造 mask ----------
        mask_lf, mask_bp, mask_hf = _radial_masks_ir(H, W, self.lf_cut, self.bp_cut, device=device, dtype=dtype)

        F_lf = F * mask_lf
        F_bp = F * mask_bp
        F_hf = F * mask_hf

        # ---------- 3) 低频增强（幅度提升 + 平滑） ----------
        A_lf = torch.abs(F_lf)
        # 使用可学习放大系数（逐批相同），并平滑滤波以防块效应
        A_lf_enh = A_lf * (1.0 + torch.relu(self.lf_gain))
        # 空域平滑：直接 IFFT 后做空域平滑会更稳定
        x_lf = torch.fft.ifft2(F_lf, norm='ortho').real
        x_lf = F.gaussian_blur(x_lf, kernel_size=[5, 5]) if hasattr(F, 'gaussian_blur') else x_lf  # optional
        # 如果系统没有 gaussian_blur，下面的 conv 会起到微调作用

        # ---------- 4) 带通：局部相位一致性判断信噪 ----------
        # 对带通部分计算相位一致性（local coherence）
        # 近似方案：对复数归一化后做小核 avgpool，coherence = |mean(norm_complex_neighbors)|
        eps = 1e-8
        Fb = F_bp
        mag = Fb.abs()
        norm_complex = Fb / (mag + eps)  # complex normalized
        # 将复数分解为 real/imag 并做平均池化近似局部统计
        real = norm_complex.real
        imag = norm_complex.imag
        # 使用 3x3 avgpool 来统计邻域相位一致性（对 real/imag 分别）
        avg_real = F.avg_pool2d(real, kernel_size=3, stride=1, padding=1)
        avg_imag = F.avg_pool2d(imag, kernel_size=3, stride=1, padding=1)
        coherence = torch.sqrt(avg_real * avg_real + avg_imag * avg_imag)  # magnitude of mean normalized complex
        # coherence ∈ [0,1], 越接近1说明相位邻域一致 -> 更可能是信号
        # soft mask: sigmoid((coherence - thresh)/smooth)
        thresh = torch.sigmoid(self.phase_thresh)  # 将阈值压在 (0,1)
        soft_mask = torch.sigmoid((coherence - thresh) / self.phase_smooth)  # (B,C,H,W)
        # 带通幅度选择性保留
        A_bp = mag
        A_bp_filtered = A_bp * (0.2 + (1.0 + F.relu(self.bp_gain)) * soft_mask)  # 非线性增益（噪声衰减）

        # ---------- 5) 高频抑制 ----------
        # 高频通过径向距离逐步抑制，并在极高频截断（防止残留噪声）
        # mask_hf 已定位高频位置，我们用径向衰减函数实现
        # 构造距离阵列，复用 mask generation from before
        ys = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)
        xs = torch.arange(W, device=device, dtype=dtype).unsqueeze(0)
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        dy = ys - cy
        dx = xs - cx
        dist = torch.sqrt(dy * dy + dx * dx)
        maxr = torch.sqrt((cy) ** 2 + (cx) ** 2)
        d = dist / (maxr + 1e-8)  # (H,W)
        d = d.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        # attenuation = exp(-gamma * (d - bp_cut)^2) for d >= bp_cut
        gamma = torch.relu(self.hf_gamma)
        att = torch.exp(-gamma * ((d - self.bp_cut) ** 2))
        # clamp extreme outer radius
        att = att * (d <= self.hf_trunc).to(dtype=dtype)
        A_hf = torch.abs(F_hf)
        A_hf_suppressed = A_hf * att

        # ---------- 6) 合并并重建 ----------
        # 使用原始相位 P 保持空间位置一致（IR 相位一般信息弱，但保持相位避免结构错位）
        # 这里把处理过的幅度组合为复谱
        # 各部分按其 mask 合并（注意维度广播）
        A_comb = (A_lf_enh * mask_lf) + (A_bp_filtered * mask_bp) + (A_hf_suppressed * mask_hf)
        # 避免复数 phase mismatch：重构 complex
        F_new = A_comb * torch.exp(1j * P)

        x_rec = torch.fft.ifft2(F_new, norm='ortho').real  # (B,C,H,W)
        # refine
        out = self.refine(x_rec)
        # residual: 保持一部分原始信息，避免过度平滑
        return out + x * 0.05
    

    
def _radial_mask(h, w, low_r, high_r, device=None, dtype=None, smooth=True):
    """
    生成径向 mask（中心为低频）。
    low_r, high_r: 半径比例，范围 0..1，相对于最大半径 (sqrt((h/2)^2+(w/2)^2))
    返回 shape (1, 1, h, w) 的 float mask
    """
    ys = torch.arange(h, device=device, dtype=dtype).unsqueeze(1)
    xs = torch.arange(w, device=device, dtype=dtype).unsqueeze(0)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    dy = ys - cy
    dx = xs - cx
    dist = torch.sqrt(dy * dy + dx * dx)
    maxr = torch.sqrt((cy) ** 2 + (cx) ** 2)
    d = dist / (maxr + 1e-8)  # 归一化到 [0,1]
    mask = ((d >= low_r) & (d < high_r)).to(dtype=dtype)
    if smooth:
        # 平滑边界，避免硬切换带来的振铃
        sigma = 0.02
        mask = torch.sigmoid((d - low_r) / sigma) * (1 - torch.sigmoid((d - high_r) / sigma))
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

###RGB
class RGBMultiFrequencyModule(nn.Module):
    """
    RGB 多频段并行频域纹理提取模块

    输入 / 输出: (B, C, H, W) -> (B, C, H, W)

    核心流程：
      1. FFT -> 得到频域 F（complex）
      2. 基于径向掩码分离低/中/高频：F_low / F_mid / F_high
      3. IFFT 回空域得到 x_low / x_mid / x_high
      4. 各自用小卷积分支处理（分别适配大核 / 常规 / 小核）
      5. 根据每个频段的能量自动生成权重并融合（softmax）
      6. 可选残差连接返回
    傅里叶变换的作用在于显式分离频段，使得网络可以对不同频段采用专门策略（增强/去噪/锐化），而不是让单一卷积去“同时”处理所有频段。
    """
    def __init__(self, channels, mid_ratio=0.25, high_ratio=0.6, reduction=8, use_residual=True):
        """
        channels: 输入通道数（输出通道数也是相同）
        mid_ratio / high_ratio: 频段划分阈值（见 _radial_mask）
        reduction: 频段权重生成 MLP 的缩放
        """
        super().__init__()
        self.C = channels
        self.mid_r = mid_ratio
        self.high_r = high_ratio
        self.use_residual = use_residual

        # 低频分支：大核，捕获全局与大尺度
        self.low_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # 中频分支：标准 3x3
        self.mid_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # 高频分支：小核 + 深度卷积，保留/增强细节
        self.high_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 频段能量到融合权重的 MLP（输入为 3 个能量标量）
        self.weight_mlp = nn.Sequential(
            nn.Linear(3, max(3, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(3, channels // reduction), 3)
        )

        # 最终融合后的微调
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # ---------- 1) FFT 到频域（复数张量） ----------
        # F: complex tensor shape (B,C,H,W)
        F = torch.fft.fft2(x, norm='ortho')  # 复数张量

        # ---------- 2) 构造径向 mask, 并分频（共享一次 FFT） ----------
        mask_low = _radial_mask(H, W, 0.0, self.mid_r, device=device, dtype=dtype)   # 低频: [0, mid_r)
        mask_mid = _radial_mask(H, W, self.mid_r, self.high_r, device=device, dtype=dtype)  # 中频: [mid_r, high_r)
        mask_high = _radial_mask(H, W, self.high_r, 1.0, device=device, dtype=dtype)  # 高频: [high_r, 1.0)

        # broadcast to (B,C,H,W) via multiplication (mask shape (1,1,H,W))
        F_low = F * mask_low
        F_mid = F * mask_mid
        F_high = F * mask_high

        # ---------- 3) IFFT 回空域并用分支处理 ----------
        x_low = torch.fft.ifft2(F_low, norm='ortho').real  # 保留实部 (B,C,H,W)
        x_mid = torch.fft.ifft2(F_mid, norm='ortho').real
        x_high = torch.fft.ifft2(F_high, norm='ortho').real

        # 分支处理（空域）
        x_low_p = self.low_branch(x_low)
        x_mid_p = self.mid_branch(x_mid)
        x_high_p = self.high_branch(x_high)

        # ---------- 4) 根据频段能量生成自适应权重 ----------
        # 以幅度能量为基础（能量 = mean(|F_band|^2)）
        def band_energy(Fb):
            mag2 = (Fb.abs() ** 2)
            # 对 batch & channel 求均值，结果形状 (B,)
            e = mag2.mean(dim=(-2, -1)).mean(dim=1)  # (B,)
            return e  # (B,)

        e_low = band_energy(F_low)   # (B,)
        e_mid = band_energy(F_mid)
        e_high = band_energy(F_high)
        E = torch.stack([e_low, e_mid, e_high], dim=1)  # (B,3)

        w = self.weight_mlp(E)  # (B,3)
        w = F.softmax(w, dim=1)  # 归一化权重
        w = w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B,3,1,1,1)

        # 重排列为 (B,1,1,1) per band per batch -> expand to channels later
        w_low = w[:, 0].view(B, 1, 1, 1)
        w_mid = w[:, 1].view(B, 1, 1, 1)
        w_high = w[:, 2].view(B, 1, 1, 1)

        # ---------- 5) 融合（按 batch 权重加权） ----------
        # 为了广播到 channels，我们 expand 到 (B,C,H,W)
        x_fused = w_low * x_low_p + w_mid * x_mid_p + w_high * x_high_p

        out = self.fuse_conv(x_fused)

        if self.use_residual:
            # 残差连接（可学习缩放由 0.1 初始）
            return out + x * 0.1
        else:
            return out
        
class RGB_HGBlock(nn.Module):
    """
    RGB-specific HGBlock with integrated multi-frequency processing.
    
    Parameters:
        c1: input channels
        cm: intermediate channels (each conv layer output)
        c2: output channels
        k: kernel size for inner convs
        n: number of inner conv layers
        lightconv: use LightConv (depthwise separable) or standard Conv
        shortcut: add residual connection (requires c1 == c2)
        act: activation function
        apply_every: apply frequency module every N layers (1=every layer, 2=every other layer)
        apply_on_concat: apply frequency module on concatenated features before output
    """
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, 
                 act=nn.ReLU(), apply_every=2, apply_on_concat=False):
        super().__init__()
        block = LightConv if lightconv else Conv
        
        # Inner conv layers (original HGBlock structure)
        self.m = nn.ModuleList(
            block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n)
        )
        
        # Squeeze and excitation convs
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2
        
        # Integrated RGB frequency module
        self.freq_module = RGBMultiFrequencyModule(channels=cm)
        self.apply_every = max(1, apply_every)
        self.apply_on_concat = apply_on_concat
    
    def forward(self, x):
        """
        前向流程：
        1. 通过 n 个卷积层提取多层特征
        2. 每隔 apply_every 层对特征进行频域多频段处理
        3. 拼接所有特征 -> squeeze -> excitation
        4. 可选：对最终输出做一次频域精炼
        """
        y = [x]
        
        for i, m in enumerate(self.m):
            feat = m(y[-1])
            
            # 在指定层应用频域处理（傅里叶变换的核心作用点）
            if i % self.apply_every == 0:
                feat = self.freq_module(feat)
            
            y.append(feat)
        
        # Standard HGBlock concat -> squeeze -> excitation
        out = self.ec(self.sc(torch.cat(y, 1)))
        
        # 可选：对整合后的特征再做一次频域处理
        if self.apply_on_concat:
            out = self.freq_module(out)
        
        return out + x if self.add else out
    

class IR_HGBlock(nn.Module):

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False,
                 act=nn.ReLU(), apply_every=2, apply_on_concat=False):
        super().__init__()
        block = LightConv if lightconv else Conv
        
        # Inner conv layers
        self.m = nn.ModuleList(
            block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n)
        )
        
        # Squeeze and excitation
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2
        
        # Integrated IR frequency filter
        self.freq_module = IRJointFilterModule(channels=cm)
        self.apply_every = max(1, apply_every)
        self.apply_on_concat = apply_on_concat
    
    def forward(self, x):
        """
        前向流程：
        1. 通过 n 个卷积层提取多层特征
        2. 每隔 apply_every 层进行频域滤波（低频增强+带通信噪分离+高频去噪）
        3. 拼接 -> squeeze -> excitation
        4. 可选：对最终输出再做一次频域滤波
        """
        y = [x]
        
        for i, m in enumerate(self.m):
            feat = m(y[-1])
            
            # 在指定层应用频域滤波（傅里叶变换的核心作用点）
            if i % self.apply_every == 0:
                feat = self.freq_module(feat)
            
            y.append(feat)
        
        # Standard HGBlock processing
        out = self.ec(self.sc(torch.cat(y, 1)))
        
        # 可选：对整合特征再做一次频域滤波
        if self.apply_on_concat:
            out = self.freq_module(out)
        
        return out + x if self.add else out
    











    ############################################################




# ==================== 辅助函数：径向频率mask (rfft2专用) ====================
def _create_radial_frequency_mask_rfft(height, width, radius_low, radius_high, device=None, dtype=None, smooth=True):
    """
    创建用于rfft2的径向频率mask（只处理正频率部分）
    
    Args:
        height, width: 特征图尺寸
        radius_low, radius_high: 频率半径范围 [0, 1]
        device, dtype: tensor 设备和类型
        smooth: 是否使用平滑过渡
    
    Returns:
        mask: shape (1, 1, H, W//2+1) - 适配rfft2输出
    """
    if dtype is None:
        dtype = torch.float32
    
    # rfft2输出形状: (H, W//2+1)
    # DC分量在(0, 0)，需要考虑频率的对称性
    
    # Y坐标：完整高度
    y_coords = torch.arange(height, device=device, dtype=dtype).unsqueeze(1)  # (H, 1)
    
    # X坐标：只有正频率部分
    x_coords = torch.arange(width // 2 + 1, device=device, dtype=dtype).unsqueeze(0)  # (1, W//2+1)
    
    # 计算中心点（DC分量位置）
    # 对于rfft2，DC在(0, 0)，但我们需要考虑频率空间的对称性
    center_y = torch.tensor(height / 2.0, device=device, dtype=dtype)
    center_x = torch.tensor(0.0, device=device, dtype=dtype)  # X方向DC在0位置
    
    # 计算Y方向的距离（考虑负频率折叠）
    y_dist = torch.minimum(y_coords, height - y_coords)
    
    # 计算X方向的距离（只有正频率）
    x_dist = x_coords
    
    # 计算归一化的径向距离
    distance = torch.sqrt(y_dist ** 2 + x_dist ** 2)
    max_radius = torch.sqrt((height / 2.0) ** 2 + (width / 2.0) ** 2)
    normalized_dist = distance / (max_radius + 1e-8)
    
    # 生成mask
    if smooth:
        sigma = 0.02
        left_edge = torch.sigmoid((normalized_dist - radius_low) / sigma)
        right_edge = 1.0 - torch.sigmoid((normalized_dist - radius_high) / sigma)
        mask = left_edge * right_edge
    else:
        mask = ((normalized_dist >= radius_low) & (normalized_dist < radius_high)).to(dtype=dtype)
    
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W//2+1)


# ==================== RGB 多频段并行纹理提取模块 (rfft2版本) ====================
class RGBMultiFrequencyModule(nn.Module):
    """
    基于rfft2的RGB多频段并行纹理提取
    
    核心思想：
    1. rfft2 将特征显式分解为低频(0-25%)、中频(25-60%)、高频(60-100%)
    2. 只处理正频率，提高计算效率
    3. 低频用大卷积核(5x5)捕获全局结构
    4. 中频用标准卷积(3x3)提取主要纹理
    5. 高频用小卷积+深度卷积保留边缘细节
    6. 基于频段能量自适应加权融合
    """
    
    def __init__(self, channels, mid_ratio=0.25, high_ratio=0.6, reduction=8, use_residual=True):
        super().__init__()
        self.channels = channels
        self.mid_ratio = mid_ratio
        self.high_ratio = high_ratio
        self.use_residual = use_residual
        
        # 低频分支：大卷积核处理全局信息
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 中频分支：标准卷积处理纹理
        self.mid_freq_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 高频分支：小卷积+深度卷积保留细节
        self.high_freq_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 频段权重生成器
        mlp_hidden = max(3, channels // reduction)
        self.weight_generator = nn.Sequential(
            nn.Linear(3, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 3),
            nn.Softmax(dim=1)
        )
        
        # 最终融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        device = x.device
        dtype = x.dtype
        
        # ========== 步骤1：rfft2到频域（只处理正频率） ==========
        fft_result = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1) 复数张量
        
        # ========== 步骤2：生成频段mask（适配rfft2输出形状） ==========
        mask_low = _create_radial_frequency_mask_rfft(height, width, 0.0, self.mid_ratio, device, dtype)
        mask_mid = _create_radial_frequency_mask_rfft(height, width, self.mid_ratio, self.high_ratio, device, dtype)
        mask_high = _create_radial_frequency_mask_rfft(height, width, self.high_ratio, 1.0, device, dtype)
        
        # ========== 步骤3：频域分离 ==========
        fft_low = fft_result * mask_low   # 低频分量
        fft_mid = fft_result * mask_mid   # 中频分量
        fft_high = fft_result * mask_high # 高频分量
        
        # ========== 步骤4：irfft2回空域（自动处理共轭对称） ==========
        spatial_low = torch.fft.irfft2(fft_low, s=(height, width), norm='ortho')
        spatial_mid = torch.fft.irfft2(fft_mid, s=(height, width), norm='ortho')
        spatial_high = torch.fft.irfft2(fft_high, s=(height, width), norm='ortho')
        
        # ========== 步骤5：各频段专用处理 ==========
        processed_low = self.low_freq_branch(spatial_low)
        processed_mid = self.mid_freq_branch(spatial_mid)
        processed_high = self.high_freq_branch(spatial_high)
        
        # ========== 步骤6：计算频段能量 ==========
        def compute_band_energy(fft_band):
            # 计算功率谱：|FFT|^2
            power = torch.abs(fft_band) ** 2
            # 在空间维度上平均，然后在通道维度上平均
            energy = power.mean(dim=(-2, -1)).mean(dim=1)  # (B,)
            return energy
        
        energy_low = compute_band_energy(fft_low)
        energy_mid = compute_band_energy(fft_mid)
        energy_high = compute_band_energy(fft_high)
        energy_vector = torch.stack([energy_low, energy_mid, energy_high], dim=1)  # (B, 3)
        
        # ========== 步骤7：生成自适应权重 ==========
        weights = self.weight_generator(energy_vector)  # (B, 3) - 已经softmax归一化
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
        
        # ========== 步骤8：加权融合 ==========
        weight_low = weights[:, 0:1]
        weight_mid = weights[:, 1:2]
        weight_high = weights[:, 2:3]
        
        fused_features = (weight_low * processed_low + 
                         weight_mid * processed_mid + 
                         weight_high * processed_high)
        
        output = self.fusion_conv(fused_features)
        
        # 可选残差连接
        if self.use_residual:
            output = output + 0.1 * x
        
        return output


# ==================== IR 低通-带通联合滤波模块 (rfft2版本) ====================
class IRJointFilterModule(nn.Module):
    """
    基于rfft2的红外联合频域滤波
    
    核心思想：
    1. rfft2 分离低频(温度主体)、带通(模糊边缘)、高频(噪声)
    2. 只处理正频率，提高效率
    3. 低频：幅度增强提升温度对比度
    4. 带通：相位一致性判断区分信号/噪声
    5. 高频：强力衰减去除传感器噪声
    6. 保持原始相位避免空间错位
    """
    
    def __init__(self, channels, lf_cut=0.2, bp_cut=0.4, hf_trunc=0.8):
        super().__init__()
        self.channels = channels
        self.lf_cut = lf_cut
        self.bp_cut = bp_cut
        self.hf_trunc = hf_trunc
        
        # 可学习的频域增益参数
        self.low_freq_gain = nn.Parameter(torch.tensor(1.2))
        self.band_pass_gain = nn.Parameter(torch.tensor(0.6))
        self.high_freq_gamma = nn.Parameter(torch.tensor(6.0))
        self.phase_threshold = nn.Parameter(torch.tensor(0.6))
        
        # 空域精炼网络
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        device = x.device
        dtype = x.dtype
        epsilon = 1e-8
        
        # ========== 步骤1：rfft2到频域 ==========
        fft_result = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1)
        global_phase = torch.angle(fft_result)  # 保存全局相位
        
        # ========== 步骤2：生成三频段mask ==========
        mask_lf = _create_radial_frequency_mask_rfft(height, width, 0.0, self.lf_cut, device, dtype)
        mask_bp = _create_radial_frequency_mask_rfft(height, width, self.lf_cut, self.bp_cut, device, dtype)
        mask_hf = _create_radial_frequency_mask_rfft(height, width, self.bp_cut, 1.0, device, dtype)
        
        fft_lf = fft_result * mask_lf
        fft_bp = fft_result * mask_bp
        fft_hf = fft_result * mask_hf
        
        # ========== 步骤3：低频增强（温度信息） ==========
        amplitude_lf = torch.abs(fft_lf)
        enhanced_amplitude_lf = amplitude_lf * (1.0 + torch.relu(self.low_freq_gain))
        
        # 使用torch.polar重构复数（更安全）
        phase_lf = torch.angle(fft_lf)
        fft_lf_enhanced = torch.polar(enhanced_amplitude_lf, phase_lf)
        
        # ========== 步骤4：带通相位一致性分析 ==========
        magnitude_bp = torch.abs(fft_bp) + epsilon
        
        # 归一化复数得到单位相量
        normalized_complex = fft_bp / magnitude_bp
        
        # 计算局部相位一致性
        # 注意：rfft2输出是(B, C, H, W//2+1)，需要适当的padding
        real_part = normalized_complex.real
        imag_part = normalized_complex.imag
        
        # 使用avg_pool2d计算局部平均（需要padding以保持尺寸）
        pooled_real = F.avg_pool2d(
            F.pad(real_part, (1, 1, 1, 1), mode='replicate'),
            kernel_size=3, stride=1, padding=0
        )
        pooled_imag = F.avg_pool2d(
            F.pad(imag_part, (1, 1, 1, 1), mode='replicate'),
            kernel_size=3, stride=1, padding=0
        )
        
        phase_coherence = torch.sqrt(pooled_real ** 2 + pooled_imag ** 2)
        
        # 基于相位一致性的软mask
        threshold = torch.sigmoid(self.phase_threshold)
        signal_mask = torch.sigmoid((phase_coherence - threshold) / 0.1)
        
        # 带通滤波：信号保留，噪声抑制
        amplitude_bp = magnitude_bp
        filtered_amplitude_bp = amplitude_bp * (0.2 + (1.0 + torch.relu(self.band_pass_gain)) * signal_mask)
        
        phase_bp = torch.angle(fft_bp)
        fft_bp_filtered = torch.polar(filtered_amplitude_bp, phase_bp)
        
        # ========== 步骤5：高频衰减（去噪） ==========
        # 创建适配rfft2的径向距离map
        y_coords = torch.arange(height, device=device, dtype=dtype).unsqueeze(1)  # (H, 1)
        x_coords = torch.arange(width // 2 + 1, device=device, dtype=dtype).unsqueeze(0)  # (1, W//2+1)
        
        # Y方向考虑负频率折叠
        y_dist = torch.minimum(y_coords, height - y_coords)
        x_dist = x_coords
        
        # 计算归一化距离
        distance = torch.sqrt(y_dist ** 2 + x_dist ** 2)
        max_radius = torch.sqrt((height / 2.0) ** 2 + (width / 2.0) ** 2)
        normalized_dist = (distance / (max_radius + epsilon)).unsqueeze(0).unsqueeze(0)
        
        # 高斯衰减滤波器
        gamma = torch.relu(self.high_freq_gamma)
        attenuation = torch.exp(-gamma * torch.clamp(normalized_dist - self.bp_cut, min=0.0) ** 2)
        attenuation = attenuation * (normalized_dist <= self.hf_trunc).to(dtype=dtype)
        
        amplitude_hf = torch.abs(fft_hf)
        suppressed_amplitude_hf = amplitude_hf * attenuation
        
        phase_hf = torch.angle(fft_hf)
        fft_hf_suppressed = torch.polar(suppressed_amplitude_hf, phase_hf)
        
        # ========== 步骤6：重建频谱 ==========
        # 方式1：分别重建后相加
        combined_fft = fft_lf_enhanced * mask_lf + fft_bp_filtered * mask_bp + fft_hf_suppressed * mask_hf
        
        # 方式2（可选）：重组幅度，保持原始相位
        # combined_amplitude = (torch.abs(fft_lf_enhanced) * mask_lf + 
        #                      torch.abs(fft_bp_filtered) * mask_bp + 
        #                      torch.abs(fft_hf_suppressed) * mask_hf)
        # combined_fft = torch.polar(combined_amplitude, global_phase)
        
        # ========== 步骤7：irfft2回空域 ==========
        reconstructed_spatial = torch.fft.irfft2(combined_fft, s=(height, width), norm='ortho')
        
        # ========== 步骤8：空域精炼 ==========
        output = self.spatial_refine(reconstructed_spatial)
        
        # 小残差连接保留原始信息
        return output + 0.05 * x


# ==================== 假设的辅助类（如果项目中不存在需要添加） ====================
class Conv(nn.Module):
    """标准卷积块"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = act if isinstance(act, nn.Module) else (nn.ReLU(inplace=True) if act else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LightConv(nn.Module):
    """轻量化卷积（深度可分离卷积）"""
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c1, k, s, k // 2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = act if isinstance(act, nn.Module) else (nn.ReLU(inplace=True) if act else nn.Identity())

    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.bn1(self.conv1(x)))))


# ==================== RGB HGBlock (rfft2版本) ====================
class RGB_HGBlock(nn.Module):
    """
    RGB专用HGBlock，集成基于rfft2的多频段并行纹理提取
    
    参数:
        c1: 输入通道数
        cm: 中间层通道数
        c2: 输出通道数
        k: 卷积核大小
        n: 内部卷积层数量
        lightconv: 是否使用轻量卷积
        shortcut: 是否使用残差连接
        act: 激活函数
        apply_every: 每隔N层应用频域处理
        apply_on_concat: 是否在concat后应用频域处理
    """
    
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, 
                 act=nn.ReLU(), apply_every=2, apply_on_concat=False):
        super().__init__()
        
        block = LightConv if lightconv else Conv
        
        # 内部卷积层
        self.conv_layers = nn.ModuleList(
            block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n)
        )
        
        # Squeeze 和 Excitation 卷积
        self.squeeze_conv = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.excitation_conv = Conv(c2 // 2, c2, 1, 1, act=act)
        self.use_shortcut = shortcut and c1 == c2
        
        # 集成RGB频域模块（rfft2版本）
        self.frequency_module = RGBMultiFrequencyModule(channels=cm)
        self.apply_every = max(1, apply_every)
        self.apply_on_concat = apply_on_concat
    
    def forward(self, x):
        feature_list = [x]
        
        # 逐层处理
        for layer_idx, conv_layer in enumerate(self.conv_layers):
            features = conv_layer(feature_list[-1])
            
            # 按间隔应用频域处理
            if layer_idx % self.apply_every == 0:
                features = self.frequency_module(features)
            
            feature_list.append(features)
        
        # 标准HGBlock处理：concat所有层特征
        concatenated = torch.cat(feature_list, dim=1)
        output = self.excitation_conv(self.squeeze_conv(concatenated))
        
        # 可选：对最终输出应用频域处理
        if self.apply_on_concat:
            output = self.frequency_module(output)
        
        # 可选残差连接
        if self.use_shortcut:
            output = output + x
        
        return output


# ==================== IR HGBlock (rfft2版本) ====================
class IR_HGBlock(nn.Module):
    """
    IR专用HGBlock，集成基于rfft2的低通-带通联合频域滤波
    
    参数:
        c1: 输入通道数
        cm: 中间层通道数
        c2: 输出通道数
        k: 卷积核大小
        n: 内部卷积层数量
        lightconv: 是否使用轻量卷积
        shortcut: 是否使用残差连接
        act: 激活函数
        apply_every: 每隔N层应用频域滤波
        apply_on_concat: 是否在concat后应用频域滤波
    """
    
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False,
                 act=nn.ReLU(), apply_every=2, apply_on_concat=False):
        super().__init__()
        
        block = LightConv if lightconv else Conv
        
        # 内部卷积层
        self.conv_layers = nn.ModuleList(
            block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n)
        )
        
        # Squeeze 和 Excitation 卷积
        self.squeeze_conv = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.excitation_conv = Conv(c2 // 2, c2, 1, 1, act=act)
        self.use_shortcut = shortcut and c1 == c2
        
        # 集成IR频域滤波模块（rfft2版本）
        self.frequency_filter = IRJointFilterModule(channels=cm)
        self.apply_every = max(1, apply_every)
        self.apply_on_concat = apply_on_concat
    
    def forward(self, x):
        feature_list = [x]
        
        # 逐层处理
        for layer_idx, conv_layer in enumerate(self.conv_layers):
            features = conv_layer(feature_list[-1])
            
            # 按间隔应用频域滤波
            if layer_idx % self.apply_every == 0:
                features = self.frequency_filter(features)
            
            feature_list.append(features)
        
        # 标准HGBlock处理：concat所有层特征
        concatenated = torch.cat(feature_list, dim=1)
        output = self.excitation_conv(self.squeeze_conv(concatenated))
        
        # 可选：对最终输出应用频域滤波
        if self.apply_on_concat:
            output = self.frequency_filter(output)
        
        # 可选残差连接
        if self.use_shortcut:
            output = output + x
        
        return output
##########
#############

# ==================== 辅助函数：径向频率mask (rfft2专用) ====================
def _create_radial_frequency_mask_rfft(height, width, radius_low, radius_high, device=None, dtype=None, smooth=True):
    if dtype is None:
        dtype = torch.float32
    
    # Y坐标：完整高度
    y_coords = torch.arange(height, device=device, dtype=dtype).unsqueeze(1)  # (H, 1)
    x_coords = torch.arange(width // 2 + 1, device=device, dtype=dtype).unsqueeze(0)  # (1, W//2+1)
    
    # 计算Y方向的距离（考虑负频率折叠）
    y_dist = torch.minimum(y_coords, height - y_coords)
    x_dist = x_coords
    
    # 计算归一化的径向距离
    distance = torch.sqrt(y_dist ** 2 + x_dist ** 2)
    
    # ✅ 修复：指定device
    max_radius = torch.sqrt(
        (torch.tensor(height, dtype=dtype, device=device) / 2.0) ** 2 +
        (torch.tensor(width, dtype=dtype, device=device) / 2.0) ** 2
    )
    normalized_dist = distance / (max_radius + 1e-8)
    
    # 生成mask
    if smooth:
        sigma = 0.02
        left_edge = torch.sigmoid((normalized_dist - radius_low) / sigma)
        right_edge = 1.0 - torch.sigmoid((normalized_dist - radius_high) / sigma)
        mask = left_edge * right_edge
    else:
        mask = ((normalized_dist >= radius_low) & (normalized_dist < radius_high)).to(dtype=dtype)
    
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W//2+1)


# ==================== RGB 多频段并行纹理提取模块 (rfft2版本) ====================
class RGBMultiFrequencyModule(nn.Module):
    """
    基于rfft2的RGB多频段并行纹理提取
    
    核心思想：
    1. rfft2 将特征显式分解为低频(0-25%)、中频(25-60%)、高频(60-100%)
    2. 只处理正频率，提高计算效率
    3. 低频用大卷积核(5x5)捕获全局结构
    4. 中频用标准卷积(3x3)提取主要纹理
    5. 高频用小卷积+深度卷积保留边缘细节
    6. 基于频段能量自适应加权融合
    """
    
    def __init__(self, channels, mid_ratio=0.25, high_ratio=0.6, reduction=8, use_residual=True):
        super().__init__()
        self.channels = channels
        self.mid_ratio = mid_ratio
        self.high_ratio = high_ratio
        self.use_residual = use_residual
        
        # 低频分支：大卷积核处理全局信息
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 中频分支：标准卷积处理纹理
        self.mid_freq_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 高频分支：小卷积+深度卷积保留细节
        self.high_freq_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 频段权重生成器
        mlp_hidden = max(3, channels // reduction)
        self.weight_generator = nn.Sequential(
            nn.Linear(3, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 3),
            nn.Softmax(dim=1)
        )
        
        # 最终融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        device = x.device
        dtype = x.dtype
        
        # ========== 步骤1：rfft2到频域（只处理正频率） ==========
        fft_result = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1) 复数张量
        
        # ========== 步骤2：生成频段mask（适配rfft2输出形状） ==========
        mask_low = _create_radial_frequency_mask_rfft(height, width, 0.0, self.mid_ratio, device, dtype)
        mask_mid = _create_radial_frequency_mask_rfft(height, width, self.mid_ratio, self.high_ratio, device, dtype)
        mask_high = _create_radial_frequency_mask_rfft(height, width, self.high_ratio, 1.0, device, dtype)
        
        # ========== 步骤3：频域分离 ==========
        fft_low = fft_result * mask_low   # 低频分量
        fft_mid = fft_result * mask_mid   # 中频分量
        fft_high = fft_result * mask_high # 高频分量
        
        # ========== 步骤4：irfft2回空域（自动处理共轭对称） ==========
        spatial_low = torch.fft.irfft2(fft_low, s=(height, width), norm='ortho')
        spatial_mid = torch.fft.irfft2(fft_mid, s=(height, width), norm='ortho')
        spatial_high = torch.fft.irfft2(fft_high, s=(height, width), norm='ortho')
        
        # ========== 步骤5：各频段专用处理 ==========
        processed_low = self.low_freq_branch(spatial_low)
        processed_mid = self.mid_freq_branch(spatial_mid)
        processed_high = self.high_freq_branch(spatial_high)
        
        # ========== 步骤6：计算频段能量 ==========
        def compute_band_energy(fft_band):
            # 计算功率谱：|FFT|^2
            power = torch.abs(fft_band) ** 2
            # 在空间维度上平均，然后在通道维度上平均
            energy = power.mean(dim=(-2, -1)).mean(dim=1)  # (B,)
            return energy
        
        energy_low = compute_band_energy(fft_low)
        energy_mid = compute_band_energy(fft_mid)
        energy_high = compute_band_energy(fft_high)
        energy_vector = torch.stack([energy_low, energy_mid, energy_high], dim=1)  # (B, 3)
        
        # ========== 步骤7：生成自适应权重 ==========
        weights = self.weight_generator(energy_vector)  # (B, 3) - 已经softmax归一化
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
        
        # ========== 步骤8：加权融合 ==========
        weight_low = weights[:, 0:1]
        weight_mid = weights[:, 1:2]
        weight_high = weights[:, 2:3]
        
        fused_features = (weight_low * processed_low + 
                         weight_mid * processed_mid + 
                         weight_high * processed_high)
        
        output = self.fusion_conv(fused_features)
        
        # 可选残差连接
        if self.use_residual:
            output = output + 0.1 * x
        
        return output


# ==================== IR 低通-带通联合滤波模块 (rfft2版本) ====================
class IRJointFilterModule(nn.Module):
    """
    基于rfft2的红外联合频域滤波
    
    核心思想：
    1. rfft2 分离低频(温度主体)、带通(模糊边缘)、高频(噪声)
    2. 只处理正频率，提高效率
    3. 低频：幅度增强提升温度对比度
    4. 带通：相位一致性判断区分信号/噪声
    5. 高频：强力衰减去除传感器噪声
    6. 保持原始相位避免空间错位
    """
    
    def __init__(self, channels, lf_cut=0.2, bp_cut=0.4, hf_trunc=0.8):
        super().__init__()
        self.channels = channels
        self.lf_cut = lf_cut
        self.bp_cut = bp_cut
        self.hf_trunc = hf_trunc
        
        # 可学习的频域增益参数
        self.low_freq_gain = nn.Parameter(torch.tensor(1.2))
        self.band_pass_gain = nn.Parameter(torch.tensor(0.6))
        self.high_freq_gamma = nn.Parameter(torch.tensor(6.0))
        self.phase_threshold = nn.Parameter(torch.tensor(0.6))
        
        # 空域精炼网络
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        device = x.device
        dtype = x.dtype
        epsilon = 1e-8
        
        # ========== 步骤1：rfft2到频域 ==========
        fft_result = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1)
        global_phase = torch.angle(fft_result)  # 保存全局相位
        
        # ========== 步骤2：生成三频段mask ==========
        mask_lf = _create_radial_frequency_mask_rfft(height, width, 0.0, self.lf_cut, device, dtype)
        mask_bp = _create_radial_frequency_mask_rfft(height, width, self.lf_cut, self.bp_cut, device, dtype)
        mask_hf = _create_radial_frequency_mask_rfft(height, width, self.bp_cut, 1.0, device, dtype)
        
        fft_lf = fft_result * mask_lf
        fft_bp = fft_result * mask_bp
        fft_hf = fft_result * mask_hf
        
        # ========== 步骤3：低频增强（温度信息） ==========
        amplitude_lf = torch.abs(fft_lf)
        enhanced_amplitude_lf = amplitude_lf * (1.0 + torch.relu(self.low_freq_gain))
        
        # 使用torch.polar重构复数（更安全）
        phase_lf = torch.angle(fft_lf)
        fft_lf_enhanced = torch.polar(enhanced_amplitude_lf, phase_lf)
        
        # ========== 步骤4：带通相位一致性分析 ==========
        magnitude_bp = torch.abs(fft_bp) + epsilon
        
        # 归一化复数得到单位相量
        normalized_complex = fft_bp / magnitude_bp
        
        # 计算局部相位一致性
        # 注意：rfft2输出是(B, C, H, W//2+1)，需要适当的padding
        real_part = normalized_complex.real
        imag_part = normalized_complex.imag
        
        # 使用avg_pool2d计算局部平均（需要padding以保持尺寸）
        pooled_real = F.avg_pool2d(
            F.pad(real_part, (1, 1, 1, 1), mode='replicate'),
            kernel_size=3, stride=1, padding=0
        )
        pooled_imag = F.avg_pool2d(
            F.pad(imag_part, (1, 1, 1, 1), mode='replicate'),
            kernel_size=3, stride=1, padding=0
        )
        
        phase_coherence = torch.sqrt(pooled_real ** 2 + pooled_imag ** 2)
        
        # 基于相位一致性的软mask
        threshold = torch.sigmoid(self.phase_threshold)
        signal_mask = torch.sigmoid((phase_coherence - threshold) / 0.1)
        
        # 带通滤波：信号保留，噪声抑制
        amplitude_bp = magnitude_bp
        filtered_amplitude_bp = amplitude_bp * (0.2 + (1.0 + torch.relu(self.band_pass_gain)) * signal_mask)
        
        phase_bp = torch.angle(fft_bp)
        fft_bp_filtered = torch.polar(filtered_amplitude_bp, phase_bp)
        
        # ========== 步骤5：高频衰减（去噪） ==========
        # 创建适配rfft2的径向距离map
        y_coords = torch.arange(height, device=device, dtype=dtype).unsqueeze(1)  # (H, 1)
        x_coords = torch.arange(width // 2 + 1, device=device, dtype=dtype).unsqueeze(0)  # (1, W//2+1)
        
        # Y方向考虑负频率折叠
        y_dist = torch.minimum(y_coords, height - y_coords)
        x_dist = x_coords
        
        # 计算归一化距离
        distance = torch.sqrt(y_dist ** 2 + x_dist ** 2)
        #max_radius = torch.sqrt((height / 2.0) ** 2 + (width / 2.0) ** 2) 

        max_radius = torch.sqrt(
                (torch.tensor(height, dtype=dtype, device=device) / 2.0) ** 2 +
                (torch.tensor(width, dtype=dtype, device=device) / 2.0) ** 2
        )
        normalized_dist = (distance / (max_radius + epsilon)).unsqueeze(0).unsqueeze(0)
        
        # 高斯衰减滤波器
        gamma = torch.relu(self.high_freq_gamma)
        attenuation = torch.exp(-gamma * torch.clamp(normalized_dist - self.bp_cut, min=0.0) ** 2)
        attenuation = attenuation * (normalized_dist <= self.hf_trunc).to(dtype=dtype)
        
        amplitude_hf = torch.abs(fft_hf)
        suppressed_amplitude_hf = amplitude_hf * attenuation
        
        phase_hf = torch.angle(fft_hf)
        fft_hf_suppressed = torch.polar(suppressed_amplitude_hf, phase_hf)
        
        # ========== 步骤6：重建频谱 ==========
        # 方式1：分别重建后相加
        combined_fft = fft_lf_enhanced * mask_lf + fft_bp_filtered * mask_bp + fft_hf_suppressed * mask_hf
        
        # 方式2（可选）：重组幅度，保持原始相位
        # combined_amplitude = (torch.abs(fft_lf_enhanced) * mask_lf + 
        #                      torch.abs(fft_bp_filtered) * mask_bp + 
        #                      torch.abs(fft_hf_suppressed) * mask_hf)
        # combined_fft = torch.polar(combined_amplitude, global_phase)
        
        # ========== 步骤7：irfft2回空域 ==========
        reconstructed_spatial = torch.fft.irfft2(combined_fft, s=(height, width), norm='ortho')
        
        # ========== 步骤8：空域精炼 ==========
        output = self.spatial_refine(reconstructed_spatial)
        
        # 小残差连接保留原始信息
        return output + 0.05 * x



# ==================== RGB HGBlock (rfft2版本) ====================
class RGB_HGBlock(nn.Module):
    """
    RGB专用HGBlock，集成基于rfft2的多频段并行纹理提取
    
    参数:
        c1: 输入通道数
        cm: 中间层通道数
        c2: 输出通道数
        k: 卷积核大小
        n: 内部卷积层数量
        lightconv: 是否使用轻量卷积
        shortcut: 是否使用残差连接
        act: 激活函数
        apply_every: 每隔N层应用频域处理
        apply_on_concat: 是否在concat后应用频域处理
    """
    
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, 
                 act=nn.ReLU(), apply_every=2, apply_on_concat=False):
        super().__init__()
        
        block = LightConv if lightconv else Conv
        
        # 内部卷积层
        self.conv_layers = nn.ModuleList(
            block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n)
        )
        
        # Squeeze 和 Excitation 卷积
        self.squeeze_conv = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.excitation_conv = Conv(c2 // 2, c2, 1, 1, act=act)
        self.use_shortcut = shortcut and c1 == c2
        
        # 集成RGB频域模块（rfft2版本）
        self.frequency_module = RGBMultiFrequencyModule(channels=cm)
        self.apply_every = max(1, apply_every)
        self.apply_on_concat = apply_on_concat
    
    def forward(self, x):
        feature_list = [x]
        
        # 逐层处理
        for layer_idx, conv_layer in enumerate(self.conv_layers):
            features = conv_layer(feature_list[-1])
            
            # 按间隔应用频域处理
            if layer_idx % self.apply_every == 0:
                features = self.frequency_module(features)
            
            feature_list.append(features)
        
        # 标准HGBlock处理：concat所有层特征
        concatenated = torch.cat(feature_list, dim=1)
        output = self.excitation_conv(self.squeeze_conv(concatenated))
        
        # 可选：对最终输出应用频域处理
        if self.apply_on_concat:
            output = self.frequency_module(output)
        
        # 可选残差连接
        if self.use_shortcut:
            output = output + x
        
        return output


# ==================== IR HGBlock (rfft2版本) ====================
class IR_HGBlock(nn.Module):
    """
    IR专用HGBlock，集成基于rfft2的低通-带通联合频域滤波
    
    参数:
        c1: 输入通道数
        cm: 中间层通道数
        c2: 输出通道数
        k: 卷积核大小
        n: 内部卷积层数量
        lightconv: 是否使用轻量卷积
        shortcut: 是否使用残差连接
        act: 激活函数
        apply_every: 每隔N层应用频域滤波
        apply_on_concat: 是否在concat后应用频域滤波
    """
    
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False,
                 act=nn.ReLU(), apply_every=2, apply_on_concat=False):
        super().__init__()
        
        block = LightConv if lightconv else Conv
        
        # 内部卷积层
        self.conv_layers = nn.ModuleList(
            block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n)
        )
        
        # Squeeze 和 Excitation 卷积
        self.squeeze_conv = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.excitation_conv = Conv(c2 // 2, c2, 1, 1, act=act)
        self.use_shortcut = shortcut and c1 == c2
        
        # 集成IR频域滤波模块（rfft2版本）
        self.frequency_filter = IRJointFilterModule(channels=cm)
        self.apply_every = max(1, apply_every)
        self.apply_on_concat = apply_on_concat
    
    def forward(self, x):
        feature_list = [x]
        
        # 逐层处理
        for layer_idx, conv_layer in enumerate(self.conv_layers):
            features = conv_layer(feature_list[-1])
            
            # 按间隔应用频域滤波
            if layer_idx % self.apply_every == 0:
                features = self.frequency_filter(features)
            
            feature_list.append(features)
        
        # 标准HGBlock处理：concat所有层特征
        concatenated = torch.cat(feature_list, dim=1)
        output = self.excitation_conv(self.squeeze_conv(concatenated))
        
        # 可选：对最终输出应用频域滤波
        if self.apply_on_concat:
            output = self.frequency_filter(output)
        
        # 可选残差连接
        if self.use_shortcut:
            output = output + x
        
        return output

################
