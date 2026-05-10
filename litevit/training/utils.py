from typing import Callable

from torch.optim.lr_scheduler import SequentialLR, LRScheduler, LinearLR
from torch.optim import Optimizer


def build_lr_scheduler_with_warmup(
    optimizer: Optimizer,
    warmup_epochs: int,
    T_max: int,
    scheduler: Callable[[Optimizer, int], LRScheduler],
) -> SequentialLR:
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1 / warmup_epochs, total_iters=warmup_epochs
    )
    main_scheduler = scheduler(optimizer, T_max = T_max - warmup_epochs)
    return SequentialLR(
        optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
    )
