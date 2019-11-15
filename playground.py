import numpy as np
import torch
from torch.nn import functional as F


test_data = np.random.randint(0, 83, (128, 100))

test_data = torch.from_numpy(test_data)
F.one_hot(test_data)
