import pytest

import torch 

def test_tiny_vit_output_shape(tiny_vit):
    x = torch.randn(2, 1, 28, 28)
    out = tiny_vit(x)
    assert out.shape == (2, 10)
    
def test_tiny_vit_batch_invariance(tiny_vit):
    x1 = torch.randn(2, 1, 28, 28)
    x2 = x1[0].unsqueeze(0)
    
    tiny_vit.eval()
    out1 = tiny_vit(x1)
    out2 = tiny_vit(x2)
    
    assert out1.shape == (2, 10)
    assert out2.shape == (1, 10)
    torch.testing.assert_close(out1[0], out2[0])

def test_tiny_vit_gradient_flow(tiny_vit):
    x = torch.randn(2, 1, 28, 28, requires_grad=True)
    out = tiny_vit(x)
    
    out.mean().backward()
    
    for name, param in tiny_vit.named_parameters():
        assert param.grad is not None, f"Gradient is None for parameter: {name}"