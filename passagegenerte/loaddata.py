from __future__ import print_function, division
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

# get batch


class TxtDataset(IterableDataset):
    def __init__(self, file_dir):
        super(TxtDataset).__init__
        self.file_dir = file_dir

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        with open(self.file_dir) as f:
            text = f.read()
        encoded = self.encoder(text)
        batch_gen = self.get_batch(encoded, self.n_seq, self.input_size)
        return iter(batch_gen)

    def encoder(self, text):
        text = set(text)
        int2char = dict(enumerate(text))
        char2int = {char: ii for ii, char in int2char.items()}
        encoded = np.array([char2int[ch] for ch in text])
        encoded = torch.from_numpy(encoded)
        encoded = F.one_hot(encoded, num_classes=len(int2char))
        return encoded

    def get_batch(self, dataset, n_seq, input_size):
        # words per batch
        batch_size = n_seq * input_size
        # number of batches
        K = len(dataset) // batch_size
        dataset = dataset[: K * batch_size]
        dataset = dataset.reshape(n_seq, -1)
        for n in range(0, dataset.shape[1], input_size):
            x = dataset[:, n:n+input_size]
            y = torch.zeros_like(x)
            y[:, :-1] = x[:, 1:]
            # next batch's first element index
            try:
                y[:, -1] = dataset[:, n+input_size]
            except IndexError:

                y[:, :-1][1:] = dataset[:, 0][1:]
            address = "data/anna.txt"
        data = DataLoader(TxtDataset(address))
        print(data)
        yield {"batch": x, "label": y}

if __name__ == "__main__":         
    address = "data/anna.txt"
    data = DataLoader(TxtDataset(address))
    print(list(data))
