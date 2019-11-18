import unittest
import numpy as np
import random

from loaddata import get_batch, chardataloader


class Testgetbatch(unittest.TestCase):
    """test get_batch to iterate all the data without Error

    """
    def test_get_batch(self):
        input_size = 100
        batch_size = 128
        n_labels = 83
        data = np.random.randint(0, n_labels, (batch_size*10*input_size+1, 1))
        gen_batch = get_batch(data, batch_size, input_size)
        x, y = next(gen_batch)
        self.assertEqual(x.shape, y.shape, "Data label should match")
    def test_chardataloader(self):
        address = "data/anna.txt"
        dataloader = chardataloader(address)
        

if __name__ == "__main__":
    unittest.main()


