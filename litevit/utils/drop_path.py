import torch 

def drop_path(x: torch.Tensor, drop_prob: float = 0.0) -> torch.Tensor:
    mask_shape: tuple[int, ...] = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    if drop_prob > 0.0:
        mask: torch.Tensor = x.new_empty(mask_shape).bernoulli_(1 - drop_prob)
        mask = mask.div_(1 - drop_prob)
        return x * mask
    return x

class DropPath(torch.nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        return drop_path(x, self.drop_prob)