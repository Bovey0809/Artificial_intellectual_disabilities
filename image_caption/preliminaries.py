import torch
import sys
import torch.utils.data as data
import nltk
from torchvision import transforms

from model import EncoderCNN, DecoderRNN
from data_loader import get_loader
sys.path.append('~/cocoapi/PythonAPI')
nltk.download('punkt')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])


vocab_threshold = 5

batch_size = 128

data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=True)

indices = data_loader.dataset.get_train_indices()
new_sampler = data.SubsetRandomSampler(indices=indices)
data_loader.batch_sampler.sampler = new_sampler
images, captions = next(iter(data_loader))


embed_size = 256
encoder = EncoderCNN(embed_size)
encoder.to(device)
images = images.to(device)
features = encoder(images)
print(features.shape)

# Specify the number of features in the hidden state of the RNN decoder.
hidden_size = 512


# Store the size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the decoder.
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move the decoder to GPU if CUDA is available.
decoder.to(device)

# Move last batch of captions (from Step 1) to GPU if CUDA is available
captions = captions.to(device)

# Pass the encoder output and captions through the decoder.
outputs = decoder(features, captions)

print('type(outputs):', type(outputs))
print('outputs.shape:', outputs.shape)

# Check that your decoder satisfies some requirements of the project! :D
assert type(
    outputs) == torch.Tensor, "Decoder output needs to be a PyTorch Tensor."
assert (outputs.shape[0] == batch_size) & (outputs.shape[1] == captions.shape[1]) & (
    outputs.shape[2] == vocab_size), "The shape of the decoder output is incorrect."
