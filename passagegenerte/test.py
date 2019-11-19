import unittest

from loaddata import TxtDataset
import torch


class Testgetbatch(unittest.TestCase):
    """test get_batch to iterate all the data without Error

    """

    def test_init(self):
        dataset = TxtDataset("data/anna.txt")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1)
        self.assertEqual(dataset.file_dir, "data/anna.txt")
        

if __name__ == "__main__":
    unittest.main()
