import torch
import torch.nn as nn
from einops import rearrange, repeat

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dilation=1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride, pad, dilation=dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class ScaleAwareBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        mid = ch // 3
        self.b1 = DepthwiseSeparableConv(ch, mid, 3, dilation=1)
        self.b2 = DepthwiseSeparableConv(ch, mid, 3, dilation=2)
        self.b3 = DepthwiseSeparableConv(ch, mid, 3, dilation=4)
        self.fuse = nn.Sequential(nn.Conv2d(mid*3, ch, 1, bias=False),
                                  nn.BatchNorm2d(ch), nn.GELU())
        self.skip = nn.Identity()

    def forward(self, x):
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)) + self.skip(x)


class CNNStream(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.GELU())
        self.stage1 = nn.Sequential(ScaleAwareBlock(64), ScaleAwareBlock(64),
                                    nn.Conv2d(64, 128, 2, 2, bias=False), nn.BatchNorm2d(128), nn.GELU())
        self.stage2 = nn.Sequential(ScaleAwareBlock(128), ScaleAwareBlock(128),
                                    nn.Conv2d(128, 256, 2, 2, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        feat = self.stage2(x)
        spatial = feat
        pooled = self.proj(self.pool(feat).flatten(1))
        return spatial, pooled


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=256):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return self.norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(dim*4, dim), nn.Dropout(0.1))
        self.drop_path_prob = drop_path

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CrossStreamAttention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.cross_attn_cv = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross_attn_vc = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_c = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.cnn_proj = nn.Linear(256, dim)

    def forward(self, cnn_spatial, vit_tokens):
        B, C, H, W = cnn_spatial.shape
        cnn_tokens = self.cnn_proj(rearrange(cnn_spatial, 'b c h w -> b (h w) c'))
        cnn_enriched, _ = self.cross_attn_cv(cnn_tokens, vit_tokens, vit_tokens)
        cnn_out = self.norm_c(cnn_tokens + cnn_enriched)
        vit_enriched, _ = self.cross_attn_vc(vit_tokens, cnn_tokens, cnn_tokens)
        vit_out = self.norm_v(vit_tokens + vit_enriched)
        return cnn_out, vit_out


class DualStreamConvViT(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=256,
                 n_heads=8, n_layers=6, num_classes=10, drop_path_rate=0.15):
        super().__init__()
        self.cnn_stream = CNNStream(embed_dim)
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.vit_blocks = nn.ModuleList([TransformerBlock(embed_dim, n_heads, dpr[i]) for i in range(n_layers)])
        self.cross_attn = CrossStreamAttention(embed_dim, n_heads)
        self.vit_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Linear(embed_dim * 3, embed_dim),
                                  nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(embed_dim, num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        cnn_spatial, _ = self.cnn_stream(x)
        vit_tokens = self.patch_embed(x)
        half = len(self.vit_blocks) // 2
        for block in self.vit_blocks[:half]:
            vit_tokens = block(vit_tokens)
        cnn_enriched, vit_tokens = self.cross_attn(cnn_spatial, vit_tokens)
        for block in self.vit_blocks[half:]:
            vit_tokens = block(vit_tokens)
        vit_tokens = self.vit_norm(vit_tokens)
        cls_token = vit_tokens[:, 0]
        vit_pooled = vit_tokens[:, 1:].mean(1)
        cnn_feat = cnn_enriched.mean(1)
        fused = torch.cat([cls_token, vit_pooled, cnn_feat], dim=-1)
        return self.head(fused)