import pytest

from litevit.utils.pos_embed import get_sincos_positional_embeddings


@pytest.mark.parametrize(
    "grid_size,embed_dim",
    [
        ((7, 7), 64),
        ((14, 14), 768),
        ((4, 8), 128),
    ],
)
def test_2d_pos_embed_output_shape(grid_size, embed_dim):
    pos_embed = get_sincos_positional_embeddings(
        grid_size=grid_size, embed_dim=embed_dim
    )
    expected = (1, grid_size[0] * grid_size[1], embed_dim)
    assert pos_embed.shape == expected


@pytest.mark.parametrize(
    "grid_size,embed_dim",
    [
        ((7, 7, 7), 48),
        ((14, 14, 10), 768),
        ((4, 8, 12), 96),
    ],
)
def test_3d_pos_embed_output_shape(grid_size, embed_dim):
    pos_embed = get_sincos_positional_embeddings(
        grid_size=grid_size, embed_dim=embed_dim
    )
    expected = (1, grid_size[0] * grid_size[1] * grid_size[2], embed_dim)
    assert pos_embed.shape == expected


@pytest.mark.parametrize(
    "grid_size",
    [
        ((7,)),
        ((7, 7, 7, 7)),
    ],
)
def test_pos_embed_invalid_grid_size(grid_size):
    with pytest.raises(ValueError):
        get_sincos_positional_embeddings(grid_size=grid_size, embed_dim=64)


@pytest.mark.parametrize(
    "grid_size, embed_dim",
    [
        ((7, 7), 65),
        ((14, 14, 10), 256),
    ],
)
def test_pos_embed_invalid_embed_dim(grid_size, embed_dim):
    with pytest.raises(ValueError):
        get_sincos_positional_embeddings(grid_size=grid_size, embed_dim=embed_dim)
