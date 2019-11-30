import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def funcname(parameter_list):
    pass


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Bilinear') != -1:
        nn.init.kaiming_uniform_(
            a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
        if m.bias:
            nn.init.zeros(tensor=m.bias)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(
            a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
        if m.bias:
            nn.init.zeros(tensor=m.bias)

    elif classname.find('BatchNorm') != -1 or classname.find('GroupNorm') != -1 or classname.find('LayerNorm') != -1:
        nn.init.uniform_(a=0, b=1, tensor=m.weight)
        nn.init.zeros(tensor=m.bias)

    elif classname.find('Cell') != -1:
        nn.init.xavier_uniform_(gain=1, tensor=m.weight_hh)
        nn.init.xavier_uniform_(gain=1, tensor=m.weight_ih)
        nn.init.ones_(tensor=m.bias_hh)
        nn.init.ones_(tensor=m.bias_ih)

    elif classname.find('RNN') != -1 or classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for w in m.all_weights:
            nn.init.xavier_uniform_(gain=1, tensor=w[2].data)
            nn.init                                                                     
            nn.init.ones_(tensor=w[0].data)
            nn.init.ones_(tensor=w[1].data)

    if classname.find('Embedding') != -1:
        nn.init.kaiming_uniform_(
            a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
