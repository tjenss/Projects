import torch
import torch.nn as nn
from torch.autograd import Variable

import string
import random
import argparse
import base64
from datetime import datetime

import os

EOS = '\n'
all_characters = string.ascii_letters + " .,;'-0123456789" + EOS
n_letters = len(all_characters)

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

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

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




    names = [line.rstrip() for line in names]
    names = (list(set(names)))
    return names

if __name__ == '__main__':

    model_path = "BoatNames.pt"
    
    model = torch.load(model_path)
    model.eval()

    cuda = False

    rand_char = random.choice(all_characters)

    with open("names.txt", "r") as n:

        training_list = n.read().splitlines()


        with open("output.txt", "a+") as f:

            name_list = f.read().splitlines()
       
            names = generate(model, prime_str=rand_char, max_names=10, cuda=cuda)

       
            for name in names:
            
                if name not in training_list:
                    print(f"{name}")
                    f.write(f"{name}\n")
                