from torchvision.datasets import FashionMNIST

from .mnist import MNISTDataModule
from litevit.data.transform_dataset import TransformDataset

class FashionMNISTDataModule(MNISTDataModule):
    def prepare_data(self):
        # Download the FashionMNIST dataset if it doesn't exist
        FashionMNIST(root=self.data_dir, train=True, download=True)
        FashionMNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            train_val_data = FashionMNIST(root=self.data_dir, train=True, download=False)
            _train_dataset, _val_dataset = self._split_dataset(
                train_val_data,
                self.split_perc[:2],  # only use the first two ratios for train/val split
            )
            self._train_dataset = TransformDataset(_train_dataset, transform=self.train_transform)
            self._val_dataset = TransformDataset(_val_dataset, transform=self.eval_transform)
            
        if stage in (None, "test"):
            test_data = FashionMNIST(root=self.data_dir, train=False, download=False)
            self._test_dataset = TransformDataset(test_data, transform=self.eval_transform)