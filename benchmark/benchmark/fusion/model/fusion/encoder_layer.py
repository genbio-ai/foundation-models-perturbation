import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        batch_first: bool,
        use_gate: bool,
        use_gelu: bool,
        xavier_init: bool,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.batch_first = batch_first
        self.use_gate = use_gate
        self.dropout = dropout
        self.use_gelu = use_gelu
        self.xavier_init = xavier_init
        
        # Manual Attention Projections to allow injecting the gate at position G1 (post-SDPA)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.ln_attn = nn.LayerNorm(embed_dim)
        self.dropout_attn = nn.Dropout(dropout)

        if self.use_gate:
            # Input is X (embed_dim), Output is Gate (embed_dim) -> Elementwise gating
            self.gate_linear = nn.Linear(embed_dim, embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU() if self.use_gelu else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        self.ln_ffn = nn.LayerNorm(embed_dim)
        self.dropout_ffn = nn.Dropout(dropout)

        if self.xavier_init:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, key_padding_mask=None):
        # x shape: (Batch, Seq, Dim) if batch_first else (Seq, Batch, Dim)
        is_batched = x.dim() == 3
        
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)


        if not self.batch_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if key_padding_mask is not None:
            sdpa_mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=x.device, dtype=q.dtype)
            sdpa_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        else:
            sdpa_mask = None
        
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=self.dropout if self.training else 0.0
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        if self.use_gate:
            # Handle batch_first difference for X
            gate_input = x if self.batch_first else x.transpose(0, 1)
            
            # Equation: Y' = Y * sigmoid(XW)
            gate = torch.sigmoid(self.gate_linear(gate_input))
            attn_out = attn_out * gate

        # Restore batch_first=False if needed for the remainder
        if not self.batch_first:
            attn_out = attn_out.transpose(0, 1)

        out = self.out_proj(attn_out)

        x = self.ln_attn(x + self.dropout_attn(out))

        ffn_x = self.ffn(x)
        x = self.ln_ffn(x + self.dropout_ffn(ffn_x))

        return x
