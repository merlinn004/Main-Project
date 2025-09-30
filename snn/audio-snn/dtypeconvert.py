import torch

class DtypeWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, dtype=torch.float16):
        self.base_dataset = base_dataset
        self.dtype = dtype
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return x.to(self.dtype), y

def convert_dataset_dtype(dataset, dtype=torch.float16):
    return DtypeWrapper(dataset, dtype)