import torch
from torch.utils.data import Dataset

eps = 1e-8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReturnDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        return x
