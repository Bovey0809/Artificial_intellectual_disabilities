import torch
from model import CharRNN


def sample(net: CharRNN, size, prime='The', top_k=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = net.predict(ch, h, top_k)
        chars.append(char)
    for ii in range(size):
        char, h = net.predict(chars[-1], h, top_k=top_k)
        chars.append(char)
    return ''.join(chars)


if __name__ == "__main__":
    with open('charrnn.net', 'rb') as f:
        checkpoint = torch.load(f)
    loaded = CharRNN(
        checkpoint['tokens'],
        n_hidden=checkpoint['n_hidden'],
        n_layers=checkpoint['n_layers'])
    loaded.load_state_dict(checkpoint['state_dict'])

    print(sample(loaded, 2000, prime='Anna', top_k=5))
