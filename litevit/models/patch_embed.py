from abc import ABC, abstractmethod

import torch 
from torch import nn
from einops import rearrange

class BasePatchEmbed(nn.Module, ABC):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int = 8,
        in_channels: int = 1,
        embed_dim: int = 192,
    ):
        super().__init__()

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.grid_size = (
            self.img_size[0] // patch_size,
            self.img_size[1] // patch_size,
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class ConvPatchEmbed(BasePatchEmbed):
    def __init__(self, img_size: int | tuple[int, int] = 224, patch_size: int = 8, in_channels: int = 1, embed_dim: int = 192):
        super().__init__(img_size, patch_size, in_channels, embed_dim)

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # check if input size is divisible by patch size
        _, _, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f'Input size ({H}x{W}) is not divisible by patch size ({self.patch_size}x{self.patch_size})')
        
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = rearrange(x, 'b d h w -> b (h w) d')  # (B, num_patches, embed_dim)
        return x