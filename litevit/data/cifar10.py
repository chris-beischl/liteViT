from torchvision.datasets import CIFAR10

from litevit.data.base import BaseDataModule
from litevit.data.transform_dataset import TransformDataset


class CIFAR10DataModule(BaseDataModule):
    def prepare_data(self):
        # Download the CIFAR10 dataset if it doesn't exist
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            train_val_data = CIFAR10(root=self.data_dir, train=True, download=False)
            _train_dataset, _val_dataset = self._split_dataset(
                train_val_data,
                self.split_perc[:2],  # only use the first two ratios for train/val split
            )
            self._train_dataset = TransformDataset(_train_dataset, transform=self.train_transform)
            self._val_dataset = TransformDataset(_val_dataset, transform=self.eval_transform)
            
        if stage in (None, "test"):
            test_data = CIFAR10(root=self.data_dir, train=False, download=False)
            self._test_dataset = TransformDataset(test_data, transform=self.eval_transform)

    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def num_channels(self) -> int:
        return 3
    
    @property
    def img_size(self) -> tuple[int, int]:
        return (32, 32)