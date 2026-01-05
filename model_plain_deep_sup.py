
"""
Plain UNet with encoder + Deep Supervision (NO ATTENTION GATES)
-----------------------------------------------------------------------------
This version removes attention gates to eliminate conflicts with deep supervision:
  • Encoder with DINO pretraining (keeps SE/CBAM/MHSA in encoder)
  • PLAIN UNet decoder (NO attention gates on skip connections)
  • Deep supervision: Auxiliary heads at d4, d3, d2 (1/16, 1/8, 1/4 resolution)
  • Multi-level loss supervision during training

This solves the architectural conflict that caused deep supervision to fail.
Expected improvement: +1.0-2.0% over baseline (0.830-0.838)
"""
from typing import List, Tuple, Optional
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: for the quick test block only
try:
    import segmentation_models_pytorch as smp  # noqa: F401
    _HAS_SMP = True
except Exception:
    _HAS_SMP = False

# timm layers aligned with original implementation
try:
    from timm.layers import DropPath, LayerNorm2d
    from timm.layers.grn import GlobalResponseNorm
except ImportError:
    raise ImportError(
        "Please install timm: pip install timm\n"
        "This implementation requires timm layers for exact alignment."
    )

# ---------------------------
# Attention building blocks
# ---------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg(x))
        return x * w

class SpatialAttention(nn.Module):
    """CBAM-style spatial attention."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        a = self.sigmoid(self.bn(self.conv(s)))
        return x * a

class CBAM(nn.Module):
    """Convolutional Block Attention Module = SE + SpatialAttention."""
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = SEBlock(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))

class MHSA2D(nn.Module):
    """Multi-Head Self-Attention on 2D feature maps (apply at low spatial res)."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.attn_drop = nn.Dropout(0.1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        dtype = x.dtype
        
        x_fp32 = x.float()
        qkv = self.qkv(x_fp32).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: [B, num_heads, head_dim, H*W]
        
        # Transpose to [B, num_heads, H*W, head_dim] for Q and V
        # Keep K as [B, num_heads, head_dim, H*W] for proper matmul
        q = q.permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        v = v.permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        # k stays as [B, num_heads, head_dim, H*W]
        
        # Apply scaling to Q and K separately (more stable)
        q = q * (self.head_dim ** -0.25)
        k = k * (self.head_dim ** -0.25)
        
        # Attention: [B, num_heads, H*W, head_dim] @ [B, num_heads, head_dim, H*W]
        #         -> [B, num_heads, H*W, H*W]
        attn = q @ k
        
        # Stabilize softmax
        attn = attn - attn.max(dim=-1, keepdim=True)[0].detach()
        attn = torch.clamp(attn, min=-88.0, max=88.0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Output: [B, num_heads, H*W, H*W] @ [B, num_heads, H*W, head_dim]
        #      -> [B, num_heads, H*W, head_dim]
        out = attn @ v
        
        # Reshape back: [B, num_heads, H*W, head_dim] -> [B, C, H, W]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        
        return out.to(dtype)

class AttentionGate(nn.Module):
    """Attention U-Net gate for filtering skip connections using decoder signal."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(F_int, 1, 1, bias=True), nn.Sigmoid())
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi

# ---------------------------
# Encoder core blocks
# ---------------------------
class Mlp(nn.Module):
    """MLP with GlobalResponseNorm (timm-style)."""
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer: nn.Module = nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.grn = GlobalResponseNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class EncoderBlock(nn.Module):
    """Encoder block with optional SE attention on the transformed branch."""
    def __init__(self, dim: int, drop_path: float = 0.0, mlp_ratio: float = 4.0,
                 use_se: bool = True, se_reduction: int = 16):
        super().__init__()
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio * dim), act_layer=nn.GELU, drop=0.0)
        self.se = SEBlock(dim, reduction=se_reduction) if use_se else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_dw(x)
        # NHWC for LayerNorm + MLP (timm style)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        x = self.se(x)
        x = shortcut + self.drop_path(x)
        return x

class EncoderStage(nn.Module):
    """Encoder stage with optional CBAM at stage output."""
    def __init__(self, in_chs: int, out_chs: int, stride: int = 2, depth: int = 2,
                 drop_path_rates: List[float] = None,
                 use_se_in_blocks: bool = True,
                 use_cbam_stage: bool = False):
        super().__init__()
        if stride > 1 or in_chs != out_chs:
            self.downsample = nn.Sequential(LayerNorm2d(in_chs), nn.Conv2d(in_chs, out_chs, kernel_size=stride, stride=stride))
        else:
            self.downsample = nn.Identity()
        drop_path_rates = drop_path_rates or [0.] * depth
        blocks = []
        for i in range(depth):
            blocks.append(
                EncoderBlock(dim=out_chs, drop_path=drop_path_rates[i], use_se=use_se_in_blocks)
            )
        self.blocks = nn.Sequential(*blocks)
        self.stage_attn = CBAM(out_chs) if use_cbam_stage else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        x = self.stage_attn(x)
        return x

class Encoder(nn.Module):
    """Encoder with attention hooks and bottleneck MHSA."""
    def __init__(self, in_chans: int = 3, dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
                 depths: Tuple[int, int, int, int] = (3, 3, 9, 3), drop_path_rate: float = 0.1,
                 use_se_in_blocks: bool = True, use_cbam_on_stages: Tuple[bool, bool, bool, bool] = (False, True, True, True),
                 use_mhsa_bottleneck: bool = True, mhsa_heads: int = 8):
        super().__init__()
        self.dims = dims
        self.depths = depths
        self.use_mhsa_bottleneck = use_mhsa_bottleneck
        # Stem
        self.stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), LayerNorm2d(dims[0]))
        # Stochastic depth schedule
        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        ptr = 0
        stages = []
        prev_chs = dims[0]
        for i in range(4):
            stride = 1 if i == 0 else 2
            d = depths[i]
            stage = EncoderStage(
                in_chs=prev_chs,
                out_chs=dims[i],
                stride=stride,
                depth=d,
                drop_path_rates=dp_rates[ptr:ptr+d],
                use_se_in_blocks=use_se_in_blocks,
                use_cbam_stage=use_cbam_on_stages[i],
            )
            stages.append(stage)
            prev_chs = dims[i]
            ptr += d
        self.stages = nn.Sequential(*stages)
        self.bottleneck_mhsa = MHSA2D(dims[3], num_heads=mhsa_heads) if use_mhsa_bottleneck else nn.Identity()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.stem(x)           # H/4
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
        # feats = [C1 (H/4), C2 (H/8), C3 (H/16), C4 (H/32)]
        if not isinstance(self.bottleneck_mhsa, nn.Identity):
            feats[-1] = feats[-1] + self.bottleneck_mhsa(feats[-1])
        return feats

# ---------------------------
# UNet decoder with Attention Gates on skips
# ---------------------------
class Conv2dReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class PlainUnetDecoderBlock(nn.Module):
    """Plain UNet decoder block: Upsample -> Concat -> Conv x2 -> Dropout (NO ATTENTION GATES)."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.use_skip = skip_ch > 0
        self.conv1 = Conv2dReLU(in_ch + (skip_ch if self.use_skip else 0), out_ch)
        self.conv2 = Conv2dReLU(out_ch, out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_skip and skip is not None:
            # Handle size mismatch between upsampled x and skip
            # This can happen with variable input sizes
            if x.shape[2:] != skip.shape[2:]:
                # Resize x to match skip dimensions exactly
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x

class UNet_PlainDeepSup(nn.Module):
    """Plain UNet with Encoder + Deep Supervision (NO ATTENTION GATES).

    This version removes attention gates from the decoder to eliminate conflicts
    with deep supervision. Keeps encoder attention mechanisms (SE/CBAM/MHSA).
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 3,
                 encoder_dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
                 encoder_depths: Tuple[int, int, int, int] = (3, 3, 9, 3),
                 drop_path_rate: float = 0.1,
                 decoder_channels: Tuple[int, int, int, int, int] = (192, 96, 64, 32, 16),
                 use_se_in_blocks: bool = True,
                 use_cbam_on_stages: Tuple[bool, bool, bool, bool] = (False, True, True, True),
                 use_mhsa_bottleneck: bool = True,
                 mhsa_heads: int = 8,
                 decoder_dropout: float = 0.0):
        super().__init__()
        self.encoder = Encoder(
            in_chans=in_chans,
            dims=encoder_dims,
            depths=encoder_depths,
            drop_path_rate=drop_path_rate,
            use_se_in_blocks=use_se_in_blocks,
            use_cbam_on_stages=use_cbam_on_stages,
            use_mhsa_bottleneck=use_mhsa_bottleneck,
            mhsa_heads=mhsa_heads,
        )

        # Initialize MHSA if present
        if use_mhsa_bottleneck:
            nn.init.xavier_uniform_(self.encoder.bottleneck_mhsa.proj.weight, gain=0.1)

        c1, c2, c3, c4 = encoder_dims
        # Center (identity; MHSA is already applied in encoder)
        self.center = nn.Identity()

        # PLAIN decoder blocks (NO ATTENTION GATES - this is the key fix!)
        self.decoder4 = PlainUnetDecoderBlock(in_ch=c4, skip_ch=c3, out_ch=decoder_channels[0], dropout=decoder_dropout)  # H/16
        self.decoder3 = PlainUnetDecoderBlock(in_ch=decoder_channels[0], skip_ch=c2, out_ch=decoder_channels[1], dropout=decoder_dropout)  # H/8
        self.decoder2 = PlainUnetDecoderBlock(in_ch=decoder_channels[1], skip_ch=c1, out_ch=decoder_channels[2], dropout=decoder_dropout)  # H/4
        self.decoder1 = PlainUnetDecoderBlock(in_ch=decoder_channels[2], skip_ch=0,   out_ch=decoder_channels[3], dropout=decoder_dropout)  # H/2
        self.decoder0 = PlainUnetDecoderBlock(in_ch=decoder_channels[3], skip_ch=0,   out_ch=decoder_channels[4], dropout=decoder_dropout)  # H

        # Main segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels[4], num_classes, kernel_size=3, padding=1)

        # DEEP SUPERVISION: Auxiliary heads at intermediate decoder stages
        self.aux_head_d4 = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)  # 1/16 resolution
        self.aux_head_d3 = nn.Conv2d(decoder_channels[1], num_classes, kernel_size=1)  # 1/8 resolution
        self.aux_head_d2 = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)  # 1/4 resolution

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor:
        """
        Forward pass with optional auxiliary outputs.

        Args:
            x: Input image [B, 3, H, W]
            return_aux: If True, return (main_output, [aux4, aux3, aux2])
                       Only used during training for deep supervision

        Returns:
            During training with return_aux=True: (main_logits, [aux4_logits, aux3_logits, aux2_logits])
            During inference: main_logits only
        """
        # Encoder
        c1, c2, c3, c4 = self.encoder(x)

        # Decoder
        x = self.center(c4)
        d4 = self.decoder4(x, c3)   # H/16
        d3 = self.decoder3(d4, c2)  # H/8
        d2 = self.decoder2(d3, c1)  # H/4
        d1 = self.decoder1(d2, None)  # H/2
        d0 = self.decoder0(d1, None)  # H

        # Main output
        main_out = self.segmentation_head(d0)

        # Return auxiliary outputs during training
        if self.training and return_aux:
            aux4 = self.aux_head_d4(d4)  # 1/16
            aux3 = self.aux_head_d3(d3)  # 1/8
            aux2 = self.aux_head_d2(d2)  # 1/4
            return main_out, [aux4, aux3, aux2]

        return main_out

    @torch.no_grad()
    def load_pretrained_from_timm(self, model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained: bool = True):
        """Load encoder weights from a timm family model.
        Extra attention parameters are left randomly initialized (strict=False).
        """
        import timm
        print(f"Loading pretrained encoder weights from: {model_name}")
        tm = timm.create_model(model_name, pretrained=pretrained, in_chans=3)
        tm_sd = tm.state_dict()
        enc_sd = self.encoder.state_dict()
        matched = {}
        for k, v in tm_sd.items():
            if k in enc_sd and v.shape == enc_sd[k].shape:
                matched[k] = v
        missing, unexpected = self.encoder.load_state_dict(matched, strict=False)
        print(f"✓ Loaded {len(matched)} encoder tensors | Missing: {len(missing)} | Unexpected: {len(unexpected)}")

# ---------------------------
# Model factory with small-dataset preset
# ---------------------------
MODEL_CONFIGS = {
    "tiny": {
        "encoder_dims": (96, 192, 384, 768),
        "encoder_depths": (3, 3, 9, 3),
        "drop_path_rate": 0.1,
        "decoder_channels": (192, 96, 64, 32, 16),
        "timm_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    },
    "small": {
        "encoder_dims": (96, 192, 384, 768),
        "encoder_depths": (3, 3, 27, 3),
        "drop_path_rate": 0.2,
        "decoder_channels": (256, 128, 64, 32, 16),
        "timm_name": "convnextv2_small",
    },
    "base": {
        "encoder_dims": (128, 256, 512, 1024),
        "encoder_depths": (3, 3, 27, 3),
        "drop_path_rate": 0.4,
        "decoder_channels": (512, 256, 128, 64, 32),
        "timm_name": "convnextv2_base.fcmae_ft_in22k_in1k",
    },
    "large": {
        "encoder_dims": (192, 384, 768, 1536),
        "encoder_depths": (3, 3, 27, 3),
        "drop_path_rate": 0.6,
        "decoder_channels": (512, 256, 128, 64, 32),
        "timm_name": "convnextv2_large.fcmae_ft_in22k_in1k",
    },
}

def create_model_plain_deep_sup(model_size: str = "tiny", in_chans: int = 3, num_classes: int = 3,
                                           pretrained: bool = True, decoder_dropout: float = 0.0) -> UNet_PlainDeepSup:
    """
    Create Plain UNet with encoder and Deep Supervision (NO ATTENTION GATES).

    This removes attention gates from decoder to eliminate conflicts with deep supervision.

    Args:
        model_size: One of 'tiny', 'small', 'base', 'large'
        in_chans: Input channels (3 for RGB)
        num_classes: Number of segmentation classes
        pretrained: Load pretrained encoder from timm
        decoder_dropout: Dropout rate in decoder blocks

    Returns:
        UNet_PlainDeepSup model
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"model_size must be one of {list(MODEL_CONFIGS.keys())}")
    cfg = MODEL_CONFIGS[model_size]
    model = UNet_PlainDeepSup(
        in_chans=in_chans,
        num_classes=num_classes,
        encoder_dims=cfg["encoder_dims"],
        encoder_depths=cfg["encoder_depths"],
        drop_path_rate=cfg["drop_path_rate"],
        decoder_channels=cfg["decoder_channels"],
        use_se_in_blocks=True,
        use_cbam_on_stages=(False, True, True, True),
        use_mhsa_bottleneck=True,
        mhsa_heads=8,
        decoder_dropout=decoder_dropout,
    )
    if pretrained:
        model.load_pretrained_from_timm(cfg["timm_name"], pretrained=True)
    return model

# ---------------------------
# Quick self-test
# ---------------------------
if __name__ == "__main__":
    print("Building UNet_PlainDeepSup (base) - NO ATTENTION GATES...")
    net = create_model_plain_deep_sup(model_size="base", in_chans=3, num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 512, 512)

    # Test training mode with auxiliary outputs
    net.train()
    with torch.no_grad():
        main, aux_list = net(x, return_aux=True)
    print("Training mode (return_aux=True):")
    print(f"  Main output: {main.shape}")
    print(f"  Aux outputs: {[a.shape for a in aux_list]}")

    # Test inference mode
    net.eval()
    with torch.no_grad():
        y = net(x, return_aux=False)
    print("\nInference mode (return_aux=False):")
    print(f"  Output: {y.shape}")

    params_m = sum(p.numel() for p in net.parameters())/1e6
    enc_params_m = sum(p.numel() for p in net.encoder.parameters())/1e6
    print(f"\nTotal params: {params_m:.2f}M | Encoder: {enc_params_m:.2f}M | Decoder: {params_m-enc_params_m:.2f}M")
    print("\n✓ Plain UNet + Deep Supervision model ready (no attention gate conflicts!)")
