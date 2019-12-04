import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)
        self.embed = nn.Linear(vocab_size, embed_size)

    def forward(self, features, captions):
        batch_size = features.shape[0]
        featuers = torch.unsqueeze(features, 1)
        captions = F.one_hot(captions, self.vocab_size)
        captions = self.embed(captions.to(torch.float))

        inputs = torch.cat((featuers, captions), dim=1)
        hidden = [torch.randn([1, batch_size, self.hidden_size]),
                  torch.randn([1, batch_size, self.hidden_size])]
        out, hidden = self.lstm(inputs, hidden)
        return out

    def sample(self, inputs, states=None, max_len=20):

        pass
