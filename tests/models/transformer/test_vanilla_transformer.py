import pytest
import torch

from litevit.models.attention import VanillaAttention
from litevit.models.block import VanillaTransformerBlock


@pytest.mark.parametrize(
    "num_tokens,embed_dim,num_heads",
    [
        (16, 64, 8),
        (32, 128, 16),
        (64, 256, 32),
    ],
)
def test_vanilla_transformer_block_output_shape(num_tokens, embed_dim, num_heads):
    attention = VanillaAttention(embed_dim=embed_dim, num_heads=num_heads)
    transformer_block = VanillaTransformerBlock(attention=attention, drop_path=0.0)
    x = torch.randn(2, num_tokens, embed_dim)  # batch_size=2
    out = transformer_block(x)

    assert out.shape == (2, num_tokens, embed_dim)


@pytest.mark.parametrize(
    "num_tokens,embed_dim,num_heads",
    [
        (16, 64, 8),
        (32, 128, 16),
        (64, 256, 32),
    ],
)
def test_vanilla_transformer_block_gradient_flow(num_tokens, embed_dim, num_heads):
    attention = VanillaAttention(embed_dim=embed_dim, num_heads=num_heads)
    transformer_block = VanillaTransformerBlock(attention=attention, drop_path=0.0)
    x = torch.randn(2, num_tokens, embed_dim, requires_grad=True)  # batch_size=2
    out = transformer_block(x)

    out.mean().backward()  # Backpropagate a simple scalar loss

    # Check if gradients are not None for all parameters
    for name, param in transformer_block.named_parameters():
        assert param.grad is not None, f"Gradient is None for parameter: {name}"
