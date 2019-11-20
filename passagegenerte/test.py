import unittest
import numpy as np


from model import CharRNN
from train import train


class Testgetbatch(unittest.TestCase):
    """test get_batch to iterate all the data without Error

    """

    def test_train(self):
        with open('data/anna.txt', 'r') as f:
            text = f.read()
        id2char = dict(enumerate(set(text)))
        char2id = {char: ii for ii, char in id2char.items()}
        chars = tuple(set(text))
        int2char = dict(enumerate(chars))
        char2int = {ch: ii for ii, ch in int2char.items()}
        encoded = np.array([char2int[ch] for ch in text])
        if 'net' in locals():
            del net
        # define and print the net
        net = CharRNN(chars, n_hidden=128, n_layers=1)
        print(net)
        n_seqs, n_steps = 128, 100
        # TRAIN
        train(net, encoded, epochs=1, n_seqs=n_seqs,
              n_steps=n_steps, lr=0.001, cuda=True, print_every=10)


if __name__ == "__main__":
    unittest.main()
