from __future__ import print_function, division
import numpy as np
from torch.utils.data import IterableDataset
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


class TxtDataset(IterableDataset):
    def __init__(self, file_dir, batch_size, seq_len):
        super(TxtDataset).__init__
        self.file_dir = file_dir        
        self.batch_size = batch_size
        self.seq_len = seq_len
        with open(self.file_dir) as f:
            self.text = f.read()
        self.n_labels = len(set(self.text))

    # def __len__(self):
    #     return len(self.batch_size * self.seq_len)

    def __getitem__(self, index):
        # encode text inot onehot
        encoded = self.encoder()
        batch = self.get_batch(encoded, index)
        return batch

    def encoder(self):
        chars = tuple(set(self.text))
        int2char = dict(enumerate(chars))
        char2int = {char: ii for ii, char in int2char.items()}
        encoded = np.array([char2int[ch] for ch in self.text])
        encoded = torch.from_numpy(encoded)
        encoded = F.one_hot(encoded, num_classes=len(int2char))
        return encoded

    def get_batch(self, dataset, index):
        words_per_batch = self.batch_size * self.seq_len
        # number of batches
        K = len(dataset) // words_per_batch
        dataset = dataset[: K * words_per_batch]
        dataset = dataset.reshape(self.batch_size, -1)
        x = dataset[:, index:index+self.seq_len]
        for n in range(0, dataset.shape[1], self.seq_len):
            x = dataset[:, n:n+self.seq_len]
            y = torch.zeros_like(x)
            y[:, :-1] = x[:, 1:]
            # next batch's first element index
            try:
                y[:, -1] = dataset[:, n+self.seq_len]
            except IndexError:
                y[:, :-1][1:] = dataset[:, 0][1:]
            yield {"batch": x, "label": y}
