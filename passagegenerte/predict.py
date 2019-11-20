import torch
from model import CharRNN


# def sample(net: CharRNN, size, prime='The', top_k=None):
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     net.to(device)
#     net.eval()
#     chars = [ch for ch in prime]
#     h = net.init_hidden(1)
#     for ch in prime:
#         char, h = net.predict(ch, h, top_k)
#     chars.append(char)
#     for ii in range(size):
#         char, h = net.predict(chars[-1], h, top_∞ok=top_k)
#         chars.append(char)
#     return ''.join(chars)
def sample(net, size, prime='The', top_k=None, cuda=True):

    if cuda:
        net.cuda()
    else:
        net.cpu()

    net.eval()

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)

    for ch in prime:
        char, h = net.predict(ch, h, cuda=cuda, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = net.predict(chars[-1], h, cuda=cuda, top_k=top_k)
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
    for _ in range(20):
        # the mean value of title is 22
        print(sample(loaded, 22, cuda=True, top_k=5, prime="中国"))
