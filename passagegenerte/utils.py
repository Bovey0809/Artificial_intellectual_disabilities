import numpy as np


def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''

    # Get the number of characters per batch
    batch_size = n_seqs * n_steps

    # TODO: Get the number of batches we can make
    n_batches = len(arr) // batch_size
    # TODO: Keep only enough characters to make full batches
    arr = arr[:batch_size * n_batches]

    # TODO: Reshape into batch_size rows
    arr = arr.reshape(n_seqs, -1)

    # TODO: Make batches
    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:, n: n + n_steps]
        y = np.zeros_like(x)
        # The targets, shifted by one
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+n_steps]
        except IndexError:
            y[:, :-1], y[:, -1][1:] = x[:, 1:], arr[:, 0][1:]
        yield x, y


def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot
