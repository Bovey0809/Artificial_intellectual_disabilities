'''
Functions:
Load data:
1. file address
'''


import numpy as np


def load_data(file_address):
    with open(file_address) as f:
        text = f.read()

    # create dict
    id2char = dict(enumerate(set(text)))
    char2id = {char: ii for ii, char in id2char.items()}
