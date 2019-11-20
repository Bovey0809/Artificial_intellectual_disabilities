import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from model import CharRNN
from utils import get_batches


def train(net: CharRNN,
          data,
          epochs=10,
          n_seqs=10,
          n_steps=50,
          lr=0.001,
          clip=5,
          val_frac=0.1,
          print_every=10,
          stop_thres=5):

    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data)*(1-val_frac))
    train_data, val_data = data[:val_idx], data[val_idx:]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net.to(device)

    # callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
    # TODO: Use earyly stop and model to rewrite the model
    # https://github.com/ncullen93/torchsample
    counter = 0
    stop_counter = 0
    early_stop = False
    for e in range(epochs):
        h = net.init_hidden(n_seqs)
        for inputs, targets in get_batches(train_data, n_seqs, n_steps):
            counter += 1
            inputs, targets = torch.from_numpy(inputs).to(
                device), torch.from_numpy(targets).to(device)
            inputs = F.one_hot(inputs, num_classes=net.n_labels)
            h = tuple([each.data for each in h])
            net.zero_grad()

            output, h = net.forward(inputs.float(), h)
            loss = criterion(output, targets.flatten())

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            if counter % print_every == 0:
                val_h = net.init_hidden(n_seq=n_seqs)
                val_losses = []
                for inputs, targets in get_batches(val_data, n_seqs, n_steps):
                    inputs = torch.from_numpy(inputs).to(device)
                    targets = torch.from_numpy(targets).to(device)
                    inputs = F.one_hot(inputs, net.n_labels)
                    output, val_h = net.forward(inputs.float(), val_h)
                    val_loss = criterion(output, targets.flatten()).item()
                    val_losses.append(val_loss)
                batch_loss = np.mean(val_losses)
                print(f"Epoch: {e+1: 2d},step: {counter: 3d},loss: {loss: .4f},val_loss: {batch_loss: .4f}")

                if abs(val_losses[-1] - batch_loss) < 0.0001:
                    stop_counter += 1
                    if stop_counter >= stop_thres:
                        print("Early Stopping")
                        early_stop = True
                        break
                    else:
                        stop_counter -= 1
                        continue
        if early_stop:
            print("STOP")
            break


if __name__ == "__main__":
    with open('data/anna.txt', 'r') as f:
        text = f.read()
    id2char = dict(enumerate(set(text)))
    char2id = {char: ii for ii, char in id2char.items()}
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    encoded = np.array([char2int[ch] for ch in text])
    if 'net' in locals():
        del net
    # define and print the net
    net = CharRNN(chars, n_hidden=512, n_layers=2)
    print(net)
    n_seqs, n_steps = 256, 128
    # TRAIN
    train(net, encoded, epochs=1000, n_seqs=n_seqs,
          n_steps=n_steps, lr=0.001, print_every=100)
    log_file = 'charrnn.net'
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'tokens': net.chars}

    with open(log_file, 'wb') as f:
        torch.save(checkpoint, f)
