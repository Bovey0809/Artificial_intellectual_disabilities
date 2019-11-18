import random
import torch
from torch import nn
from torch.nn import functional as F


class CharRNN(nn.Module):
    """Character level RNN for generate text from text

    """

    def __init__(self, tokens, n_steps=100, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.001):
        super(CharRNN, self).__init__()
        self.tokens = tokens
        self.n_steps = n_steps
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.lr = lr

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {char: i for i, char in self.int2char.items()}
        self.n_labels = len(self.chars)

        self.lstm = nn.LSTM(self.n_labels, self.n_hidden, self.n_layers,
                            dropout=self.drop_prob, batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)

        self.fc = nn.Linear(self.n_hidden, self.n_labels)
        # Use different weights to init fc layer.
        self.initweight()

    def forward(self, x, hc):
        x, [h, c] = self.lstm(x, hc)
        x = self.dropout(x)

        x = x.view(-1, self.hidden)

        x = self.fc(x)

        return x, (h, c)

    def init_hidden(self, n_seq):
        # init hidden shape is [num_layers, num_batches, hidden_dim]
        # Get the parameters's type from model
        weights = next(self.parameters()).data
        # return hidden state and cell state
        return (weights.new(self.n_layers, n_seq, self.n_hidden),
                weights.new(self.n_layers, n_seq, self.n_hidden))

    def initweight(self):
        self.fc.bias.fill_(0)
        self.fc.weight.uniform_(-1, 1)

    def predict(self, char, h=None, top_k=None):
        # given a char predict next char
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = self.char2int[char]
        x = F.one_hot(torch.tensor(x, dtype=torch.long), self.n_labels)
        # n_seq = 1

        x = x.view(1, 1, self.n_labels)
        x.to(device)

        hc = self.init_hidden(1)
        x, hc = self.forward(x, hc)
        p = F.softmax(x, dim=1)
        if top_k:
            p, indexes = torch.topk(p, dim=1)
            p, indexes = p.numpy().squeeze(), indexes.numpy().squeeze()
            index = random.choices(indexes, p=p / p.sum())
        else:
            index = p.argmax().item()
        char = self.int2char[index.numpy().squeeze()]
        return char, hc
