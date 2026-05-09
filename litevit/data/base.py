from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from lightning import LightningDataModule


class BaseDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        train_transform: nn.Module | None = None,
        eval_transform: nn.Module | None  = None,
        split_perc: tuple[float, float, float] = (0.8, 0.1, 0.1),
        num_workers: int = 0,
        batch_size: int = 32,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        data_dir: str = "./data",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.train_transform = train_transform or nn.Identity()
        self.eval_transform = eval_transform or nn.Identity()
        self.split_perc = split_perc
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.data_dir = data_dir
        self.seed = seed

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        
    @property
    @abstractmethod
    def num_classes(self) -> int: 
        ... 
        
    @property
    @abstractmethod
    def num_channels(self) -> int: 
        ...
        
    @property
    @abstractmethod
    def img_size(self) -> tuple[int, int]: 
        ... 

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage=None):
        pass
    
    def _split_dataset(self, dataset: Dataset, split_ratio: tuple[float, ...]) -> tuple[Dataset, ...]:
        return random_split(
            dataset,
            split_ratio,
            generator=torch.Generator().manual_seed(self.seed),
        )
    
    def _make_dataloader(self, dataset, shuffle: bool) -> DataLoader: 
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None, 
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader(self._test_dataset, shuffle=False)