from torch.utils.data import Dataset
import torch

class TransformDataset(Dataset):
    def __init__(self, dataset, transform: torch.nn.Module):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        return self.transform(x), y