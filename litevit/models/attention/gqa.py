import torch
import torch.nn as nn
from einops import rearrange

from .base import BaseAttention


class GroupedQueryAttention(BaseAttention):
    """based on paper:
        'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
        Checkpoints'
        Ainslie et al., 2023

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads. Must divide embed_dim evenly.
        num_kv_heads: Number of groups for kv heads.
        dropout: Dropout probability applied to attention weights and output
        projection.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)

        self.head_dim = embed_dim // num_heads
        self.num_kv_heads = num_kv_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if self.num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads \
                    ({num_kv_heads})"
            )

        # scale factor for attention scores of each head
        self.scale = 1 / (self.head_dim**0.5)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * num_kv_heads * self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head attention using einops
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_kv_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_kv_heads)

        # Repeating k, v heads
        k = torch.repeat_interleave(k, self.num_heads // self.num_kv_heads, dim=1)
        v = torch.repeat_interleave(v, self.num_heads // self.num_kv_heads, dim=1)

        # Compute attention scores using einsum for efficient batch matrix
        # multiplication
        # qk shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        qk = (q @ k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            qk = qk + mask

        attn = torch.softmax(qk, dim=-1)
        attn = self.attn_dropout(attn)

        # Compute attention output using einsum
        # attn_v shape: (batch_size, num_heads, seq_len_q, head_dim)
        attn_v = attn @ v

        # Reshape back to (batch_size, seq_len_q, embed_dim)
        attn_v = rearrange(attn_v, "b h s d -> b s (h d)")

        out = self.proj_dropout(self.out_proj(attn_v))

        attn_weights = attn if return_attention else None
        return out, attn_weights
