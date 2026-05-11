import torch
from torch import nn

from ..attention import BaseAttention
from .base import BaseTransformerBlock


class VanillaTransformerBlock(BaseTransformerBlock):
    def __init__(
        self,
        attention: BaseAttention,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
        norm: type[nn.Module] | str = nn.LayerNorm,
    ):
        super().__init__(attention, mlp_ratio, dropout, drop_path, activation, norm)

        self.norm1 = self.norm(self.embed_dim)
        self.norm2 = self.norm(self.embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * mlp_ratio)),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(int(self.embed_dim * mlp_ratio), self.embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x), mask=mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x