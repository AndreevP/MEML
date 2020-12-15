import torch
from torch.utils.data import Dataset

class NpDataset(Dataset):

    def __init__(self, data):
        self.data = data[0]
        self.labels = data[1]
    
    def __getitem__(self, i):
        return self.data[i], self.labels[i]
    
    def __len__(self):
        return len(self.data)