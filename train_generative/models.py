"""
DiT (Diffusion Transformer) for 1D data and the DiffusionAutoEncoder wrapper.
Adopted from: https://github.com/facebookresearch/DiT
"""

import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.vision_transformer import Attention, Mlp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def modulate(x, shift, scale):
    """Adaptive layer normalization modulation.  x: (B, T, D), shift/scale: (B, D)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed(embed_dim, length, max_period=10000):
    """Sinusoidal positional embedding of shape (length, embed_dim)."""
    position = np.arange(length, dtype=np.float32)
    half_dim = embed_dim // 2
    freqs = np.exp(-math.log(max_period) * np.arange(half_dim, dtype=np.float32) / half_dim)
    out = np.einsum("i,j->ij", position, freqs)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    if embed_dim % 2 == 1:
        emb = np.concatenate([emb, np.zeros([length, 1])], axis=1)
    return emb


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.linear1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.nonlinear = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.norm1(self.linear1(t_freq))
        t_emb = self.nonlinear(t_emb)
        t_emb = self.norm2(self.linear2(t_emb))
        return t_emb


# ---------------------------------------------------------------------------
# 1-D Input Embedding
# ---------------------------------------------------------------------------

class SequenceEmbed(nn.Module):
    """Projects (B, C, T) -> (B, T, hidden_size)."""

    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.relu(self.norm1(self.linear1(x)))
        x = self.relu(self.norm2(self.linear2(x)))
        return x


# ---------------------------------------------------------------------------
# DiT Blocks
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        z_emb_dim=None,
        z_emb_dropout=0.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
        mlp_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-5)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attention_dropout,
            proj_drop=projection_dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-5)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=mlp_dropout,
        )

        if z_emb_dim is None:
            z_emb_dim = hidden_size

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.z_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(z_emb_dim, 6 * hidden_size))
        self.z_dropout = nn.Dropout(z_emb_dropout)

    def forward(self, x, c, z_emb=None):
        shift_c, scale_c, gate_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.adaLN_modulation(c).chunk(6, dim=1)

        if z_emb is not None:
            z_mod = self.z_dropout(self.z_adaLN_modulation(z_emb))
            shift_z, scale_z, gate_z, shift_mlp_z, scale_mlp_z, gate_mlp_z = z_mod.chunk(6, dim=1)
            shift_msa = shift_c + shift_z
            scale_msa = scale_c + scale_z
            gate_msa = gate_c + gate_z
            shift_mlp = shift_mlp_c + shift_mlp_z
            scale_mlp = scale_mlp_c + scale_mlp_z
            gate_mlp = gate_mlp_c + gate_mlp_z
        else:
            shift_msa, scale_msa, gate_msa = shift_c, scale_c, gate_c
            shift_mlp, scale_mlp, gate_mlp = shift_mlp_c, scale_mlp_c, gate_mlp_c

        # MSA
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final projection with adaptive modulation."""

    def __init__(self, hidden_size, out_channels, z_emb_dim=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-5)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        if z_emb_dim is None:
            z_emb_dim = hidden_size
        self.z_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(z_emb_dim, 2 * hidden_size))

    def forward(self, x, c, z_emb=None):
        shift_c, scale_c = self.adaLN_modulation(c).chunk(2, dim=1)
        if z_emb is not None:
            shift_z, scale_z = self.z_adaLN_modulation(z_emb).chunk(2, dim=1)
            shift, scale = shift_c + shift_z, scale_c + scale_z
        else:
            shift, scale = shift_c, scale_c
        return self.linear(modulate(self.norm_final(x), shift, scale))


# ---------------------------------------------------------------------------
# DiT Model
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """
    Diffusion Transformer for 1D data (B, C, T).

    Parameters
    ----------
    input_size : int
        Sequence length T.
    in_channels : int
        Number of input channels C.
    hidden_size : int
        Transformer hidden dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    z_emb_dim : int, optional
        Dimension of the conditioning embedding.
    out_channels : int, optional
        Output channels (defaults to in_channels).
    """

    def __init__(
        self,
        input_size=50,
        in_channels=1,
        hidden_size=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        cond_drop_prob=0.0,
        z_emb_dim=None,
        attention_dropout=0.0,
        projection_dropout=0.0,
        mlp_dropout=0.0,
        z_emb_dropout=0.0,
        out_channels=None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = in_channels
        self.self_condition = False
        self.input_size = input_size
        self.cond_drop_prob = cond_drop_prob

        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        if z_emb_dim is None:
            z_emb_dim = hidden_size

        self.x_embedder = SequenceEmbed(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size), requires_grad=False)
        self.num_tokens = input_size

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, z_emb_dim=z_emb_dim,
                attention_dropout=attention_dropout, projection_dropout=projection_dropout,
                mlp_dropout=mlp_dropout, z_emb_dropout=z_emb_dropout,
            )
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, out_channels, z_emb_dim=z_emb_dim)
        self.null_z_emb = nn.Parameter(torch.randn(z_emb_dim))

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_tokens)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.x_embedder.linear1.weight, std=0.02)
        nn.init.normal_(self.x_embedder.linear2.weight, std=0.02)
        nn.init.normal_(self.x_embedder.norm1.weight, std=0.02)
        nn.init.normal_(self.x_embedder.norm2.weight, std=0.02)

        nn.init.normal_(self.t_embedder.linear1.weight, std=0.02)
        nn.init.normal_(self.t_embedder.linear2.weight, std=0.02)
        nn.init.normal_(self.t_embedder.norm1.weight, std=0.02)
        nn.init.normal_(self.t_embedder.norm2.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.z_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.z_adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.z_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.z_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, z_emb=None, x_self_cond=None, cond_drop_prob=None):
        """
        x: (B, C, T), t: (B,), z_emb: (B, D) or None.
        Returns: (B, out_channels, T).
        """
        batch, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        t = t.expand(batch) if t.ndim == 0 else t

        x = self.x_embedder(x) + self.pos_embed  # (B, T, D)
        c = self.t_embedder(t)  # (B, D)

        if cond_drop_prob > 0 and z_emb is not None:
            keep = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            null = repeat(self.null_z_emb, "d -> b d", b=batch)
            z_emb = torch.where(rearrange(keep, "b -> b 1"), z_emb, null)

        for block in self.blocks:
            x = block(x, c, z_emb=z_emb)

        x = self.final_layer(x, c, z_emb=z_emb)  # (B, T, out_channels)
        return x.transpose(1, 2)  # (B, out_channels, T)


# ---------------------------------------------------------------------------
# Autoencoder Wrapper
# ---------------------------------------------------------------------------

class DiffusionAutoEncoder(nn.Module):
    """
    Thin wrapper that pairs an (identity) autoencoder with a DiT diffusion model.

    For the PCA latent mode the autoencoder is unused (identity), but this
    wrapper keeps the same interface as the original celldiff code.
    """

    def __init__(self, diffusion_model, unconditional=False):
        super().__init__()
        self.diffusion = diffusion_model
        self.channels = diffusion_model.channels
        self.self_condition = False
        self.unconditional = unconditional

    def forward(self, x, t, z_emb=None, x_self_cond=None):
        if self.unconditional:
            z_emb = None
        return self.diffusion(x, t, z_emb=z_emb)
