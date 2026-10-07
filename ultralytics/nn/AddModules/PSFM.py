import math
import torch.nn as nn
import torch

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True) -> None:
        super().__init__()
        
        self.dwconv = DWConv(c1, c1, 3)
        self.pwconv = Conv(c1, c2, 1)
    
    def forward(self, x):
        return self.pwconv(self.dwconv(x))

class GEFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(GEFM, self).__init__()
        self.RGB_K= DSConv(out_C, out_C, 3)
        self.RGB_V = DSConv(out_C, out_C, 3)
        self.Q = DSConv(in_C, out_C, 3)
        self.INF_K= DSConv(out_C, out_C, 3)
        self.INF_V = DSConv(out_C, out_C, 3)       
        self.Second_reduce = DSConv(in_C, out_C, 3)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, y):
        Q = self.Q(torch.cat([x,y], dim=1))
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width*height)
        RGB_K = RGB_K.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width*height)
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height,width)
        RGB_refine = self.gamma1*RGB_refine+y
        
        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width*height)
        INF_K = INF_K.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width*height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height,width) 
        INF_refine = self.gamma2 * INF_refine + x
        
        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))        
        return out 

class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=2):
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(DSConv(mid_C * i, mid_C, 3))

        self.fuse = DSConv(in_C + mid_C, out_C, 3)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for i in self.denseblock:
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)

class PSFM(nn.Module):
    def __init__(self, Channel):
        super(PSFM, self).__init__()
        self.RGBobj = DenseLayer(Channel, Channel)
        self.Infobj = DenseLayer(Channel, Channel)           
        self.obj_fuse = GEFM(Channel * 2, Channel)
        
    def forward(self, data):
        rgb, depth = data
        rgb_sum = self.RGBobj(rgb)        
        Inf_sum = self.Infobj(depth)        
        out = self.obj_fuse(rgb_sum,Inf_sum)
        return out
    


class GEFM_rev(nn.Module):
    """
    改进点：
    1. V用1x1保真投影
    2. RGB的K用3x3提取纹理
    3. IR的K用1x1 + 全局池化提取热区
    4. 添加temperature缩放
    """
    def __init__(self, in_C, out_C):
        super(GEFM_rev, self).__init__()
        
        # ========== K的设计：针对模态特性 ==========
        
        # RGB的K: 3x3卷积捕捉纹理细节
        self.RGB_K = DSConv(out_C, out_C, 3)
        
        # IR的K: 1x1 + 全局池化捕捉热区
        self.INF_K_channel = nn.Conv2d(out_C, out_C, 1, bias=False)
        self.INF_K_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_C, out_C // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_C // 16, out_C, 1),
            nn.Sigmoid()
        )
        
        # ========== V的设计：1x1保真投影 ==========
        self.RGB_V = nn.Conv2d(out_C, out_C, 1, bias=False)
        self.INF_V = nn.Conv2d(out_C, out_C, 1, bias=False)
        
        # ========== Q的设计：保持原样 ==========
        self.Q = DSConv(in_C, out_C, 3)
        
        # ========== 其他组件 ==========
        self.Second_reduce = DSConv(in_C, out_C, 3)
        self.gamma1 = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(1) * 0.1)
        self.temperature = nn.Parameter(torch.ones(1))  # 可学习温度
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, y):
        Q = self.Q(torch.cat([x, y], dim=1))
        m_batchsize, C, height, width = x.size()
        
        # ========== RGB -> INF 注意力 ==========
        RGB_K = self.RGB_K(x)  # 3x3纹理提取
        RGB_V = self.RGB_V(x)  # 1x1保真
        
        RGB_V = RGB_V.view(m_batchsize, -1, width*height)
        RGB_K = RGB_K.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width*height)
        
        # 添加temperature缩放
        RGB_mask = torch.bmm(RGB_K, RGB_Q) / (self.temperature.clamp(min=0.1) + 1e-6)
        RGB_mask = self.softmax(RGB_mask)
        
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)
        RGB_refine = self.gamma1 * RGB_refine + y
        
        # ========== INF -> RGB 注意力 ==========
        # 1x1通道融合 + 全局热区加权
        INF_K_local = self.INF_K_channel(y)
        INF_K_global = self.INF_K_global(y)
        INF_K = INF_K_local * INF_K_global  # 热区加权
        
        INF_V = self.INF_V(y)  # 1x1保真
        
        INF_V = INF_V.view(m_batchsize, -1, width*height)
        INF_K = INF_K.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width*height)
        
        INF_mask = torch.bmm(INF_K, INF_Q) / (self.temperature.clamp(min=0.1) + 1e-6)
        INF_mask = self.softmax(INF_mask)
        
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width) 
        INF_refine = self.gamma2 * INF_refine + x
        
        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))        
        return out




class PSFM_rev(nn.Module):
    def __init__(self, Channel):
        super(PSFM_rev, self).__init__()
        self.RGBobj = DenseLayer(Channel, Channel)
        self.Infobj = DenseLayer(Channel, Channel)
        
        # ✅ 使用改进的GEFM
        self.obj_fuse = GEFM_rev(Channel * 2, Channel)
        
    def forward(self, data):
        rgb, depth = data
        rgb_sum = self.RGBobj(rgb)        
        Inf_sum = self.Infobj(depth)        
        out = self.obj_fuse(rgb_sum, Inf_sum)
        return out
    
##############################

class PolarizedLinearCrossAttention(nn.Module):
    """
    极化线性交叉注意力模块
    核心创新：
    1. 将交叉注意力从 O(N²) 降到 O(N)
    2. 引入极化机制增强表达能力
    3. 可学习的非线性强度
    """
    def __init__(self, in_channels, out_channels, num_heads=4, alpha=4, eps=1e-6):
        super().__init__()
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.alpha = alpha
        self.eps = eps
        
        # Q/K/V 投影
        self.q_proj = DSConv(in_channels, out_channels, 3)
        self.k_proj = DSConv(out_channels, out_channels, 3)
        self.v_proj = DSConv(out_channels, out_channels, 3)
        
        # 极化注意力的可学习参数
        self.power = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, 1, 1, out_channels))
        
        # 残差权重
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 激活函数
        self.kernel_function = nn.ReLU()
        
    def forward(self, x_source, x_target):
        """
        Args:
            x_source: 源模态特征 (B, C_out, H, W) - 提供 K, V
            x_target: 目标模态特征 (B, C_in, H, W) - 提供 Q
        Returns:
            增强后的目标模态特征 (B, C_out, H, W)
        """
        B, C_out, H, W = x_source.shape
        N = H * W
        
        # 1. 生成 Q, K, V
        Q = self.q_proj(torch.cat([x_source, x_target], dim=1))  # (B, C_out, H, W)
        K = self.k_proj(x_source)  # (B, C_out, H, W)
        V = self.v_proj(x_source)  # (B, C_out, H, W)
        
        # Reshape to (B, num_heads, N, head_dim)
        Q = Q.view(B, self.num_heads, self.head_dim, N).transpose(2, 3)  # (B, h, N, d)
        K = K.view(B, self.num_heads, self.head_dim, N).transpose(2, 3)  # (B, h, N, d)
        V = V.view(B, self.num_heads, self.head_dim, N).transpose(2, 3)  # (B, h, N, d)
        
        # 2. 缩放（稳定训练）
        scale = nn.functional.softplus(self.scale).view(1, self.num_heads, 1, self.head_dim)
        Q = Q / (scale + self.eps)
        K = K / (scale + self.eps)
        
        # 3. 可学习的 power 参数
        power = 1 + self.alpha * torch.sigmoid(self.power)  # (1, h, 1, 1)
        
        # 4. 极化分解（核心创新）
        Q_pos = self.kernel_function(Q) ** power    # (B, h, N, d)
        Q_neg = self.kernel_function(-Q) ** power   # (B, h, N, d)
        K_pos = self.kernel_function(K) ** power    # (B, h, N, d)
        K_neg = self.kernel_function(-K) ** power   # (B, h, N, d)
        
        # 5. 拼接成极化特征
        Q_sim = torch.cat([Q_pos, Q_neg], dim=-1)  # (B, h, N, 2d) - 相似路径
        Q_opp = torch.cat([Q_neg, Q_pos], dim=-1)  # (B, h, N, 2d) - 对立路径
        K_all = torch.cat([K_pos, K_neg], dim=-1)  # (B, h, N, 2d)
        
        # 6. 线性注意力计算（O(N) 复杂度）
        # 关键: 先计算 K^T @ V，再与 Q 相乘
        V1, V2 = torch.chunk(V, 2, dim=-1)  # 分成两半
        
        # 相似路径
        KV_sim = torch.matmul(K_all.transpose(-2, -1), V1) * (N ** -0.5)  # (B, h, 2d, d/2)
        Z_sim = torch.matmul(Q_sim, K_all.mean(dim=2, keepdim=True).transpose(-2, -1))  # 归一化
        Z_sim = Z_sim + self.eps
        out_sim = torch.matmul(Q_sim, KV_sim) / Z_sim  # (B, h, N, d/2)
        
        # 对立路径
        KV_opp = torch.matmul(K_all.transpose(-2, -1), V2) * (N ** -0.5)  # (B, h, 2d, d/2)
        Z_opp = torch.matmul(Q_opp, K_all.mean(dim=2, keepdim=True).transpose(-2, -1))
        Z_opp = Z_opp + self.eps
        out_opp = torch.matmul(Q_opp, KV_opp) / Z_opp  # (B, h, N, d/2)
        
        # 7. 融合两路输出
        out = torch.cat([out_sim, out_opp], dim=-1)  # (B, h, N, d)
        
        # 8. Reshape 回原始形状
        out = out.transpose(2, 3).contiguous().view(B, C_out, H, W)
        
        # 9. 残差连接
        out = self.gamma * out + x_target
        
        return out


class PolarizedLinearGEFM(nn.Module):
    """
    极化线性全局增强融合模块
    相比原始 GEFM:
    - 复杂度从 O(N²) 降到 O(N)
    - 增加极化机制
    - 保持双向融合特性
    """
    def __init__(self, in_C, out_C, num_heads=4, alpha=4):
        super().__init__()
        
        # 极化线性交叉注意力（双向）
        self.rgb_attention = PolarizedLinearCrossAttention(
            in_channels=in_C, 
            out_channels=out_C, 
            num_heads=num_heads,
            alpha=alpha
        )
        
        self.inf_attention = PolarizedLinearCrossAttention(
            in_channels=in_C, 
            out_channels=out_C, 
            num_heads=num_heads,
            alpha=alpha
        )
        
        # 最终融合
        self.fuse = DSConv(in_C, out_C, 3)
        
        print(f'Polarized Linear GEFM initialized: in_C={in_C}, out_C={out_C}, heads={num_heads}, alpha={alpha}')
        
    def forward(self, x, y):
        """
        Args:
            x: RGB 特征 (B, C, H, W)
            y: INF/Depth 特征 (B, C, H, W)
        Returns:
            融合后的特征 (B, C_out, H, W)
        """
        # 双向极化线性交叉注意力
        # RGB 用 INF 信息增强
        rgb_refine = self.rgb_attention(x_source=x, x_target=y)
        
        # INF 用 RGB 信息增强  
        inf_refine = self.inf_attention(x_source=y, x_target=x)
        
        # 融合
        out = self.fuse(torch.cat([rgb_refine, inf_refine], dim=1))
        
        return out


class PL_PSFM(nn.Module):
    """
    使用极化线性 GEFM 的 PSFM 模块
    """
    def __init__(self, Channel, num_heads=4, alpha=4):
        super().__init__()
        self.RGBobj = DenseLayer(Channel, Channel)
        self.Infobj = DenseLayer(Channel, Channel)
        
        # 使用极化线性 GEFM 替换原始 GEFM
        self.obj_fuse = PolarizedLinearGEFM(Channel * 2, Channel, num_heads=num_heads, alpha=alpha)
        
    def forward(self, data):
        rgb, depth = data
        rgb_sum = self.RGBobj(rgb)
        inf_sum = self.Infobj(depth)
        out = self.obj_fuse(rgb_sum, inf_sum)
        return out

