# %%
import sys
sys.path.append(
    '/home/houbowei/Artificial_intellectual_disabilities/image_caption')
sys.path.append('~/cocoapi/PythonAPI')
from model import EncoderCNN, DecoderRNN

import os
import torch
from data_loader import get_loader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt


transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])
# %%
data_loader = get_loader(transform_test, mode='test',
                         vocab_file='/home/houbowei/Artificial_intellectual_disabilities/image_caption/vocab.pkl')
orig_image, image = next(iter(data_loader))
plt.imshow(np.squeeze(orig_image))
plt.title('example image')
plt.show()

# %%
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

encoder_file = './models/256-512-512-encoder-1.pkl'
decoder_file = './models/256-512-512-decoder-1.pkl'

embed_size = 512
hidden_size = 512

vocab_size = len(data_loader.dataset.vocab)

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

encoder.eval()
decoder.eval()

encoder.load_state_dict(torch.load(encoder_file))
decoder.load_state_dict(torch.load(decoder_file))

encoder.to(device)
decoder.to(device)

# Move image Pytorch Tensor to GPU if CUDA is available.
image = image.to(device)

# Obtain the embedded image features.
features = encoder(image).unsqueeze(1)

# Pass the embedded image features through the model to get a predicted caption.
output = decoder.sample(features)
print('example output:', output)

assert (type(output) == list), "Output needs to be a Python list"
assert all([type(x) == int for x in output]
           ), "Output should be a list of integers."
assert all([x in data_loader.dataset.vocab.idx2word for x in output]
           ), "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."
