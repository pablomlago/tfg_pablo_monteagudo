import torch
from torch.utils.data import Dataset

class   TracesDatasetSeqTime(Dataset):
    def __init__(self, X, X_valid_len, Y, Y_valid_len):
        self._X = X
        self._X_valid_len = X_valid_len
        self._Y = Y
        self._Y_valid_len = Y_valid_len
    def __len__(self):
        return len(self._X)
    def __getitem__(self, idx):
        #X, X_valid_len, Y, Y_valid_len
        return (torch.tensor(self._X[idx]).permute(1,0),
                torch.tensor(self._X_valid_len[idx]),
                torch.tensor(self._Y[idx]),
                torch.tensor(self._Y_valid_len[idx]))