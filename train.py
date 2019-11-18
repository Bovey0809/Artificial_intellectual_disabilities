import torch
from torch import nn
from torch.nn import functional
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from model import CharRNN


def train(net, dataloader, epochs=10, n_seqs=10, n_steps=50, lr=0.001, clip=5, val_frac=0.1, cuda=Flase, print_every=10):
    '''Training a network
    Arguments:

    '''
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    