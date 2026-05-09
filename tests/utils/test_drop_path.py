import pytest 

import torch

from litevit.utils.drop_path import DropPath, drop_path

def test_drop_path_no_drop(): 
    x = torch.randn(2, 3)
    out = drop_path(x, drop_prob=0.0)
    torch.testing.assert_close(x, out)

@pytest.mark.parametrize("drop_prob,training", [(0.0, False), (0.0, True), (0.5, False)])
def test_DropPath_no_drop(drop_prob, training):
    x = torch.randn(2, 3)
    drop_path_layer = DropPath(drop_prob=drop_prob)
    drop_path_layer.train(training)
    out = drop_path_layer(x)
    torch.testing.assert_close(x, out)
