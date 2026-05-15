from typing import Any, Protocol

import hydra
import torch
from omegaconf import DictConfig
from torch import nn

from ..utils.pos_embed import get_sincos_positional_embeddings
from .block import BaseTransformerBlock
from .patch_embed import BasePatchEmbed


class BlockFactory(Protocol):
    def __call__(self, drop_path: float) -> BaseTransformerBlock: ...


class ViT(nn.Module):
    """Vision Transformer with optional register tokens.

    Register tokens are additional learnable tokens appended to the sequence
    after the CLS token. Motivated by Darcet et al., "Vision Transformers Need
    Registers" (2023) — registers give the model dedicated slots for global
    information, reducing attention artifacts in patch tokens and improving
    representation quality. Set num_registers=0 (default) to disable.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed: BasePatchEmbed,
        block_factory: BlockFactory,
        depth: int = 12,
        num_classes: int = 2,
        num_registers: int = 0,
        pos_embed_type: str = "sincos",
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.patch_embed = patch_embed
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.pos_embed_type = pos_embed_type

        self.pos_drop = nn.Dropout(p=dropout)

        # Store class token as a parameter, since it is learnable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_registers, self.embed_dim)
            )

        if pos_embed_type == "sincos":
            # Register the sincos positional embedding as a buffer since it is fixed
            # and not learnable
            self.pos_embed: torch.Tensor
            self.register_buffer(
                "pos_embed",
                get_sincos_positional_embeddings(
                    grid_size=self.patch_embed.grid_size,
                    embed_dim=self.embed_dim,
                ),
            )
        else:
            raise NotImplementedError(
                f"Positional embedding type '{pos_embed_type}' not implemented"
            )

        # create stochastic depth:
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # construct transformer blocks using the provided block factory and drop path
        # rates
        self.blocks = nn.ModuleList(
            [block_factory(drop_path=dpr[i]) for i in range(depth)]
        )

        # final layer normalization before classification head
        self.norm = nn.LayerNorm(self.embed_dim)

        # classification head
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # initialize wieghts
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.num_registers > 0:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # add positional embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, : x.shape[1], :]  # (B, num_patches, embed_dim)

        # add CLS token to the beginning of the sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        if self.num_registers > 0:
            register_tokens = self.register_tokens.expand(B, -1, -1)
            x = torch.cat((x, register_tokens), dim=1)

        # apply dropout to the input of the transformer blocks
        x = self.pos_drop(x)

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # apply final layer normalization
        x = self.norm(x)

        # take the CLS token output for classification
        cls_output = x[:, 0]  # (B, embed_dim)
        logits: torch.Tensor = self.head(cls_output)  # (B, num_classes)
        return logits


def build_vit(
    embed_dim: int,
    patch_embed_cfg: DictConfig,
    attention_cfg: DictConfig,
    block_cfg: DictConfig,
    depth: int,
    num_classes: int,
    num_registers: int = 0,
    pos_embed_type: str = "sincos",
    dropout: float = 0.0,
    drop_path_rate: float = 0.0,
    **kwargs: dict[Any, Any],
) -> ViT:
    patch_embed = hydra.utils.instantiate(patch_embed_cfg)

    def block_factory(drop_path: float) -> BaseTransformerBlock:
        attention = hydra.utils.instantiate(attention_cfg)
        block: BaseTransformerBlock = hydra.utils.instantiate(
            block_cfg, attention=attention, drop_path=drop_path
        )
        return block

    return ViT(
        embed_dim=embed_dim,
        patch_embed=patch_embed,
        block_factory=block_factory,
        depth=depth,
        num_classes=num_classes,
        num_registers=num_registers,
        pos_embed_type=pos_embed_type,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
    )
