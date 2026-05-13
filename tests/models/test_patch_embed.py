import pytest
import torch

from litevit.models.patch_embed import ConvPatchEmbed


@pytest.mark.parametrize(
    "img_size,patch_size,in_channels,embed_dim",
    [
        (224, 8, 1, 192),
        (256, 16, 3, 384),
        (128, 8, 1, 128),
        (28, 4, 1, 64),
    ],
)
def test_conv_patch_embed_output_shape(img_size, patch_size, in_channels, embed_dim):
    patch_embed = ConvPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )
    x = torch.randn(2, in_channels, img_size, img_size)  # batch_size=2
    out = patch_embed(x)

    num_patches = (img_size // patch_size) ** 2

    expected_shape = (2, num_patches, embed_dim)
    assert out.shape == expected_shape


@pytest.mark.parametrize(
    "img_size,patch_size,in_channels,embed_dim",
    [
        (224, 6, 1, 192),
        (256, 12, 3, 384),
        (128, 6, 1, 128),
        (28, 6, 1, 64),
    ],
)
def test_conv_patch_embed_invalid_input_size(
    img_size, patch_size, in_channels, embed_dim
):
    patch_embed = ConvPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )
    x = torch.randn(2, in_channels, img_size, img_size)  # batch_size=2
    with pytest.raises(ValueError):
        patch_embed(x)


def test_conv_patch_embed_gradient_flow():
    img_size = 224
    patch_size = 8
    in_channels = 1
    embed_dim = 192

    patch_embed = ConvPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )
    x = torch.randn(2, in_channels, img_size, img_size, requires_grad=True)
    out = patch_embed(x)

    out.mean().backward()

    for name, param in patch_embed.named_parameters():
        assert param.grad is not None, f"Gradient is None for parameter: {name}"
