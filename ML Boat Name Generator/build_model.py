from __future__ import division, print_function, unicode_literals

import glob
import math
import random
import string
import time
import unicodedata
from io import open

import numpy as np
import torch
import torch.nn as nn
import unidecode
from torch.autograd import Variable
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
## File path
file_name = "names.txt"

EOS = '\n'
all_characters = string.ascii_letters + " .,;'-0123456789" + EOS
n_letters = len(all_characters) # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_characters
    )

# Read a file and split into lines
def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Build the lines file
lines = open(file_name, encoding='utf-8').read().strip().split('\n')

file, file_len = read_file(file_name)
all_names = [unicodeToAscii(line) for line in lines]

n_names = len(all_names)

if n_names == 0:
    raise RuntimeError('Data not found.')

print(f'# names: {n_names}, # chars: {n_letters}')
print("\n...")
for i in range(10):
    print(f"{all_names[np.random.randint(n_names)]}")

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    
    random.shuffle(all_names)
    dataset = '\n'.join(s for s in all_names)
    
    for bi in range(batch_size):
        start_index = random.randint(0, len(dataset) - chunk_len - 1)
        end_index = start_index + chunk_len + 1
        
        chunk = dataset[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    return inp, target

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

### Network Parameters ###
n_input_chars = n_letters
n_layers = 3
hidden_size = 128

decoder = CharRNN(
    n_input_chars,
    hidden_size,
    n_letters,
    n_layers=n_layers,
)

learning_rate = 0.0001

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

#criterion = nn.NLLLoss()
### Training Parameters ###
print_every = 60
n_epochs = 3400
chunk_len = 250
batch_size = 32
cuda = False

save_filename = "BoatNames.pt"

def train(inp, target):
    hidden = decoder.init_hidden(batch_size)
    
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data / chunk_len

def save():
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

max_length = 20

def generate(decoder, prime_str='A', max_names = 15, max_length = 100, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    names = []
    
    while len(names) < max_names:
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        top_i = max(0, min(top_i, len(all_characters)-1))

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char

        if (predicted_char == '\n' and len(predicted) > 0):
            names.append(predicted)
            predicted = ''
        
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()
    
    return ''.join(names)

start = time.time()
all_losses = []
loss_avg = 0

try:
    if cuda:
        decoder.cuda()
    
    print("Training for %d epochs..." % n_epochs)
    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = train(*random_training_set(chunk_len, batch_size))
        loss_avg += loss
        all_losses.append(loss)

        if epoch % print_every == 0:
            print('[%s (%d %d%%) loss = %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
            rand_char = random.choice(all_characters)
            print(generate(decoder, prime_str=rand_char, max_names=10, cuda=cuda), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()