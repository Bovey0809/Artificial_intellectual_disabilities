import torchvision.datasets as dset
from torchvision import transforms
import nltk
import torch
import torchtext
from torch.utils.data import DataLoader
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform_train = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


def target_transform(textlist):
    results = []
    for text in textlist:
        tokens = ['<start>']
        text = nltk.word_tokenize(text.lower())
        tokens.extend(text)
        tokens.append('<end>')
        results.append(tokens)
    return results

cap = dset.CocoCaptions(root='/home/houbowei/cocoapi/images/train2014',
                        annFile='/home/houbowei/cocoapi/annotations/captions_train2014.json',
                        transform=transform_train,
                        target_transform=target_transform)

print('Number of samples: ', len(cap))
img, target = cap[3]  # load 4th sample

print("Image Size: ", img.size())
print(target)

sampler = torch.utils.data.Sampler(data_source)



def get_loader(transform, root, annFile, mode='train', batch_size=256, vocab_threshold=None, vocab_file='./vocab.pkl', unk_word="<unk>"):
    pass