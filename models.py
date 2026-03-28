"""PyTorch deep learning architectures for CFRP property prediction."""

import numpy as np

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

if TORCH_AVAILABLE:

    class ResidualBlock(nn.Module):
        def __init__(self, in_dim, out_dim, dropout=0.15):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
                nn.SiLU(), nn.Dropout(dropout))
            self.skip = (nn.Linear(in_dim, out_dim, bias=False)
                         if in_dim != out_dim else nn.Identity())

        def forward(self, x):
            return self.block(x) + self.skip(x)

    class ResidualMLP(nn.Module):
        def __init__(self, input_dim, layers, dropout=0.15):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, layers[0]),
                nn.BatchNorm1d(layers[0]), nn.SiLU())
            blocks = []
            for i in range(len(layers)-1):
                blocks.append(ResidualBlock(layers[i], layers[i+1], dropout))
            self.trunk = nn.Sequential(*blocks)
            self.head = nn.Linear(layers[-1], 1)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.head(self.trunk(self.input_proj(x))).squeeze(-1)

    class FeatureAttention(nn.Module):
        def __init__(self, embed_dim, n_heads=4):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=n_heads,
                batch_first=True, dropout=0.1)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x):
            x_3d = x.unsqueeze(-1)
            pad_size = self.attn.embed_dim - 1
            if pad_size > 0:
                x_3d = torch.nn.functional.pad(x_3d, (0, pad_size))
            attn_out, _ = self.attn(x_3d, x_3d, x_3d)
            return self.norm(attn_out + x_3d).mean(dim=-1)

    class AttentionMLP(nn.Module):
        def __init__(self, input_dim, layers, dropout=0.15, n_heads=4):
            super().__init__()
            self.embed_dim = max(n_heads, (input_dim // n_heads) * n_heads)
            self.attention = FeatureAttention(self.embed_dim, n_heads)
            self.proj_in = (nn.Linear(input_dim, self.embed_dim)
                            if input_dim != self.embed_dim else nn.Identity())
            self.proj_out = (nn.Linear(self.embed_dim, input_dim)
                             if input_dim != self.embed_dim else nn.Identity())
            self.residual_mlp = ResidualMLP(input_dim, layers, dropout)

        def forward(self, x):
            x_proj = self.proj_in(x) if not isinstance(self.proj_in, nn.Identity) else x
            x_attn = self.attention(x_proj)
            x_back = self.proj_out(x_attn) if not isinstance(self.proj_out, nn.Identity) else x_attn
            return self.residual_mlp(x + x_back)

    class MultiHeadCFRPNet(nn.Module):
        def __init__(self, input_dim, trunk_layers, head_dim=32, dropout=0.15):
            super().__init__()
            self.trunk = ResidualMLP(input_dim, trunk_layers, dropout)
            self.trunk.head = nn.Identity()
            last_dim = trunk_layers[-1]
            self.strength_head = nn.Sequential(
                nn.Linear(last_dim, head_dim), nn.SiLU(), nn.Linear(head_dim, 1))
            self.modulus_head = nn.Sequential(
                nn.Linear(last_dim, head_dim), nn.SiLU(), nn.Linear(head_dim, 1))

        def forward(self, x):
            shared = self.trunk(x)
            return self.strength_head(shared).squeeze(-1), self.modulus_head(shared).squeeze(-1)


def build_pytorch_model(model_type, input_dim, config):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    layers = config["layers"]
    dropout = config.get("dropout", 0.15)
    if model_type == "residual":
        return ResidualMLP(input_dim, layers, dropout)
    elif model_type == "attention":
        return AttentionMLP(input_dim, layers, dropout, n_heads=4)
    elif model_type == "multihead":
        return MultiHeadCFRPNet(input_dim, layers, dropout=dropout)
    raise ValueError(f"Unknown model type: {model_type}")
