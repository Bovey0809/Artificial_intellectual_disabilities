from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

# get batch


def get_batch(dataset, n_seq, input_size):
    # words per batch
    batch_size = n_seq * input_size
    # number of batches
    K = len(dataset) // batch_size
    dataset = dataset[: K * batch_size]
    dataset = dataset.reshape(n_seq, -1)
    for n in range(0, dataset.shape[1], input_size):
        x = dataset[:, n:n+input_size]
        y = np.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        # next batch's first element index
        try:
            y[:, -1] = dataset[:, n+input_size]
        except IndexError:
            y[:, :-1][1:] = dataset[:, 0][1:]
        yield x, y


def encodedtxt(txtfileaddress):
    with open(txtfileaddress) as f:
        text = f.read()
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    encoded = np.array([char2int[ch] for ch in text])


    