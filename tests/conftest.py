import pytest
import torch

from litevit.models.attention.vanilla import VanillaAttention
from litevit.models.block import VanillaTransformerBlock
from litevit.models.patch_embed import ConvPatchEmbed
from litevit.models.vit import ViT


@pytest.fixture
def tiny_vit():
    embed_dim = 64
    num_heads = 2
    depth = 4

    img_size = 28
    patch_size = 4
    in_channels = 1

    patch_embed = ConvPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )

    def block_factory(drop_path):
        return VanillaTransformerBlock(
            attention=VanillaAttention(
                embed_dim=embed_dim, num_heads=num_heads, dropout=0.0
            ),
            drop_path=drop_path,
        )

    return ViT(
        embed_dim=embed_dim,
        patch_embed=patch_embed,
        block_factory=block_factory,
        depth=depth,
        num_classes=10,
    )


@pytest.fixture(params=["cpu", "mps"])
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    return torch.device(request.param)
