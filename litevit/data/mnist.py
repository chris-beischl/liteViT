import torch
from torch.utils.data import random_split

from torchvision.datasets import MNIST

from litevit.data.base import BaseDataModule
from litevit.data.transform_dataset import TransformDataset


class MNISTDataModule(BaseDataModule):
    def prepare_data(self):
        # Download the MNIST dataset if it doesn't exist
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        
        if stage in (None, "fit"):
            train_val_data = MNIST(root=self.data_dir, train=True, download=False)
            self._train_dataset, self._val_dataset = self._split_dataset(
                train_val_data
            )
        if stage in (None, "test"):
            test_data = MNIST(root=self.data_dir, train=False, download=False)
            self._test_dataset = TransformDataset(test_data, transform=self.eval_transform)

    def _split_dataset(self, dataset: torch.utils.data.Dataset) -> tuple[TransformDataset, TransformDataset]:
        train_perc = self.split_perc[0]
        val_perc = self.split_perc[1]

        train_ratio = train_perc / (train_perc + val_perc)
        val_ratio = 1.0 - train_ratio

        train_set, val_set = random_split(
            dataset,
            [train_ratio, val_ratio],
            generator=torch.Generator().manual_seed(self.seed),
        )

        train_set = TransformDataset(train_set, transform=self.train_transform)
        val_set = TransformDataset(val_set, transform=self.eval_transform)

        return train_set, val_set

    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def num_channels(self) -> int:
        return 1
    
    @property
    def img_size(self) -> tuple[int, int]:
        return (28, 28)