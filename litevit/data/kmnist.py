from torchvision.datasets import KMNIST

from .mnist import MNISTDataModule
from litevit.data.transform_dataset import TransformDataset

class KMNISTDataModule(MNISTDataModule):
    def prepare_data(self):
        # Download the KMNIST dataset if it doesn't exist
        KMNIST(root=self.data_dir, train=True, download=True)
        KMNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        if stage in (None, "fit"):
            train_val_data = KMNIST(root=self.data_dir, train=True, download=False)
            self._train_dataset, self._val_dataset = self._split_dataset(
                train_val_data
            )
        if stage in (None, "test"):
            test_data = KMNIST(root=self.data_dir, train=False, download=False)
            self._test_dataset = TransformDataset(test_data, transform=self.eval_transform)