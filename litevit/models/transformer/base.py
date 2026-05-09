from abc import ABC, abstractmethod

import torch
from torch import nn

from ..attention.base import BaseAttention
from ...utils.drop_path import DropPath

class BaseTransformerBlock(nn.Module, ABC):
    def __init__(
        self,
        attention: BaseAttention,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
        norm: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.attention = attention
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.drop_path = DropPath(drop_path)
        self.norm = norm
        self.activation = activation

        self.embed_dim = attention.embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor: ...
