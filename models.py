"""
models.py — Deep Learning Architectures for CFRP Property Prediction
=====================================================================
Implements PyTorch-based neural networks with residual connections,
batch normalisation, feature-level self-attention, and multi-task
heads.  Falls back gracefully to sklearn MLP if PyTorch is unavailable.

Architecture Summary
--------------------
  ResidualBlock    : Dense → BatchNorm → SiLU → Dropout + skip connection
  ResidualMLP      : Stacked residual blocks with adaptive pooling
  AttentionMLP     : Self-attention over features before residual prediction
  MultiHeadCFRPNet : Shared trunk + dual heads (strength + modulus)
"""

import numpy as np

# ── Check PyTorch availability ───────────────────────────────────────────────
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

if TORCH_AVAILABLE:

    class ResidualBlock(nn.Module):
        """
        Pre-activation residual block for tabular data.

        Architecture:  Linear → BatchNorm → SiLU → Dropout  (+skip)

        Skip connection uses a linear projection if input and output
        dimensions differ, otherwise identity.  This follows the
        strategy of He et al. (2016) adapted for MLP architectures.
        """
        def __init__(self, in_dim, out_dim, dropout=0.15):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            # Projection shortcut (1×1 linear) when dimensions change
            self.skip = (nn.Linear(in_dim, out_dim, bias=False)
                         if in_dim != out_dim else nn.Identity())

        def forward(self, x):
            return self.block(x) + self.skip(x)


    class ResidualMLP(nn.Module):
        """
        Deep residual MLP for material property regression.

        Stacks N residual blocks with decreasing width, followed by
        a single linear output head.  Kaiming initialisation ensures
        stable gradient flow at depth.

        Parameters
        ----------
        input_dim : int   — number of input features
        layers    : list  — hidden layer widths, e.g. [256, 128, 64]
        dropout   : float — dropout probability per block
        """
        def __init__(self, input_dim, layers, dropout=0.15):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, layers[0]),
                nn.BatchNorm1d(layers[0]),
                nn.SiLU(),
            )
            blocks = []
            for i in range(len(layers) - 1):
                blocks.append(ResidualBlock(layers[i], layers[i+1], dropout))
            self.trunk = nn.Sequential(*blocks)
            self.head  = nn.Linear(layers[-1], 1)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.trunk(x)
            return self.head(x).squeeze(-1)


    class FeatureAttention(nn.Module):
        """
        Scaled dot-product self-attention over feature dimensions.

        Treats each feature as a "token" and computes attention weights
        to identify feature interactions — particularly relevant for
        composites where certain feature combinations (e.g., Vf × E_f)
        have non-linear synergistic effects.

        Inspired by TabNet (Arik & Pfister, 2019) and FT-Transformer
        (Gorishniy et al., 2021) adapted for small tabular datasets.
        """
        def __init__(self, embed_dim, n_heads=4):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=n_heads,
                batch_first=True, dropout=0.1)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x):
            # x: (batch, features) → (batch, features, 1) → expand
            # Treat each feature as a token with 1-dim embedding
            x_3d = x.unsqueeze(-1)  # (B, F, 1)
            # Pad to embed_dim
            pad_size = self.attn.embed_dim - 1
            if pad_size > 0:
                x_3d = torch.nn.functional.pad(x_3d, (0, pad_size))
            attn_out, _ = self.attn(x_3d, x_3d, x_3d)
            attn_out = self.norm(attn_out + x_3d)
            # Pool back to (B, F)
            return attn_out.mean(dim=-1)


    class AttentionMLP(nn.Module):
        """
        Feature-attention augmented MLP.

        Pipeline: FeatureAttention → ResidualMLP

        The attention layer learns which feature interactions matter
        most for the prediction, providing interpretability alongside
        improved accuracy on heterogeneous composite datasets.
        """
        def __init__(self, input_dim, layers, dropout=0.15, n_heads=4):
            super().__init__()
            # Embedding dimension must be divisible by n_heads
            self.embed_dim = max(n_heads, (input_dim // n_heads) * n_heads)
            self.attention = FeatureAttention(self.embed_dim, n_heads)
            self.proj_in   = nn.Linear(input_dim, self.embed_dim) \
                             if input_dim != self.embed_dim else nn.Identity()
            self.proj_out  = nn.Linear(self.embed_dim, input_dim) \
                             if input_dim != self.embed_dim else nn.Identity()
            self.residual_mlp = ResidualMLP(input_dim, layers, dropout)

        def forward(self, x):
            x_proj = self.proj_in(x) if not isinstance(
                self.proj_in, nn.Identity) else x
            x_attn = self.attention(x_proj)
            x_back = self.proj_out(x_attn) if not isinstance(
                self.proj_out, nn.Identity) else x_attn
            # Additive residual from attention
            x_enhanced = x + x_back
            return self.residual_mlp(x_enhanced)


    class MultiHeadCFRPNet(nn.Module):
        """
        Multi-task architecture with shared trunk and dual prediction heads
        for tensile strength and modulus.

        Shared-trunk multi-task learning exploits the physical correlation
        between strength and stiffness in composites (both depend on
        Vf, fibre properties, and orientation), enabling implicit
        regularisation and improved generalisation.
        """
        def __init__(self, input_dim, trunk_layers, head_dim=32, dropout=0.15):
            super().__init__()
            # Shared feature extractor
            self.trunk = ResidualMLP(input_dim, trunk_layers, dropout)
            # Override trunk's head — we need the penultimate output
            self.trunk.head = nn.Identity()
            last_dim = trunk_layers[-1]
            # Task-specific heads
            self.strength_head = nn.Sequential(
                nn.Linear(last_dim, head_dim), nn.SiLU(),
                nn.Linear(head_dim, 1),
            )
            self.modulus_head = nn.Sequential(
                nn.Linear(last_dim, head_dim), nn.SiLU(),
                nn.Linear(head_dim, 1),
            )

        def forward(self, x):
            shared = self.trunk(x)
            strength = self.strength_head(shared).squeeze(-1)
            modulus  = self.modulus_head(shared).squeeze(-1)
            return strength, modulus


def build_pytorch_model(model_type, input_dim, config):
    """
    Factory function to create PyTorch model from config.

    Parameters
    ----------
    model_type : str  — "residual", "attention", or "multihead"
    input_dim  : int  — number of input features
    config     : dict — must contain 'layers' and 'dropout'

    Returns
    -------
    model : nn.Module
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    layers  = config["layers"]
    dropout = config.get("dropout", 0.15)

    if model_type == "residual":
        return ResidualMLP(input_dim, layers, dropout)
    elif model_type == "attention":
        return AttentionMLP(input_dim, layers, dropout, n_heads=4)
    elif model_type == "multihead":
        return MultiHeadCFRPNet(input_dim, layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
