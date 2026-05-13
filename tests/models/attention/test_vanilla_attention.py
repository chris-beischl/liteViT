import pytest
import torch

from litevit.models.attention import VanillaAttention


@pytest.mark.parametrize(
    "num_tokens,embed_dim,num_heads",
    [
        (16, 64, 8),
        (32, 128, 16),
        (64, 256, 32),
    ],
)
def test_vanilla_attention_output_shape(num_tokens, embed_dim, num_heads):
    attention = VanillaAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(2, num_tokens, embed_dim)  # batch_size=2
    out, attn_weights = attention.forward(x, return_attention=True)

    assert out.shape == (2, num_tokens, embed_dim)
    assert attn_weights.shape == (2, num_heads, num_tokens, num_tokens)


def test_vanilla_attention_no_return_attention():
    embed_dim = 64
    num_heads = 2
    num_tokens = 16
    attention = VanillaAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(2, num_tokens, embed_dim)  # batch_size=2
    out, attn_weights = attention.forward(x, return_attention=False)

    assert out.shape == (2, num_tokens, embed_dim)
    assert attn_weights is None


@pytest.mark.parametrize(
    "num_tokens,embed_dim,num_heads",
    [
        (16, 64, 8),
        (32, 128, 16),
        (64, 256, 32),
    ],
)
def test_attention_weights_sum_to_one(num_tokens, embed_dim, num_heads):
    attention = VanillaAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(2, num_tokens, embed_dim)  # batch_size=2
    _, attn_weights = attention(x, return_attention=True)

    # Sum of attention weights across the last dimension should be 1
    attn_sum = attn_weights.sum(dim=-1)
    torch.testing.assert_close(attn_sum, torch.ones_like(attn_sum))


def test_num_heads_not_divisible_by_embed_dim():
    with pytest.raises(ValueError):
        VanillaAttention(embed_dim=64, num_heads=7)
