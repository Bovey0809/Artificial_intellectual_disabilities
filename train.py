import torch
from torch import nn
from torch.nn import functional as F


def train(net, data, epochs=10, batch_size=16, seq_len=128, lr=0.0001, clip=5, val_frac=0.1, print_every=10):
