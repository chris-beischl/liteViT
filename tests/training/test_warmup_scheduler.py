from functools import partial

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR

from litevit.training.utils import build_lr_scheduler_with_warmup


def test_returns_sequential_lr():
    model = torch.nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    optimizer.step()
    scheduler = partial(CosineAnnealingLR, eta_min=0.0)

    warmup_scheduler = build_lr_scheduler_with_warmup(
        optimizer, warmup_epochs=20, t_max=100, scheduler=scheduler
    )
    assert isinstance(warmup_scheduler, SequentialLR)


@pytest.mark.parametrize("base_lr", [0.01, 0.001, 0.0001])
@pytest.mark.parametrize("warmup_epochs", [2, 10, 20])
def test_warmup_reaches_base_lr(base_lr, warmup_epochs):
    model = torch.nn.Linear(2, 2)
    optimizer = AdamW(model.parameters(), lr=base_lr)
    scheduler = partial(CosineAnnealingLR, eta_min=0.0)

    warmup_scheduler = build_lr_scheduler_with_warmup(
        optimizer, warmup_epochs=warmup_epochs, t_max=100, scheduler=scheduler
    )

    for _ in range(warmup_epochs):
        optimizer.step()
        warmup_scheduler.step()

    torch.testing.assert_close(optimizer.param_groups[0]["lr"], base_lr)


@pytest.mark.parametrize("base_lr", [0.01, 0.001, 0.0001])
@pytest.mark.parametrize("warmup_epochs", [2, 10, 20])
def test_lr_strictly_decreases_after_warmup(base_lr, warmup_epochs):
    model = torch.nn.Linear(2, 2)
    optimizer = AdamW(model.parameters(), lr=base_lr)
    scheduler = partial(CosineAnnealingLR, eta_min=0.0)

    warmup_scheduler = build_lr_scheduler_with_warmup(
        optimizer, warmup_epochs=warmup_epochs, t_max=100, scheduler=scheduler
    )

    for _ in range(warmup_epochs):
        optimizer.step()
        warmup_scheduler.step()

    last_lr = optimizer.param_groups[0]["lr"]

    for _ in range(warmup_epochs, 100):
        optimizer.step()
        warmup_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr < last_lr
        last_lr = current_lr
