from typing import Any

import torch
from torch.utils.data import Dataset


class TransformDataset(Dataset[tuple[Any, Any]]):
    def __init__(
        self, dataset: Dataset[tuple[Any, Any]], transform: torch.nn.Module
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        x, y = self.dataset[idx]
        return self.transform(x), y
