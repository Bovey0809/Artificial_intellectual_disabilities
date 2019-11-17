import numpy as np


# one hot


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