"""
Containing the RNN class
@author: Sindre Andre Jacobsen
"""
from torch import nn
import torch
import math
torch.manual_seed(0)


class MyRNN(nn.Module):
    def __init__(self, input_size,  hidden_dim, total_intents, int2intent):
        super(MyRNN, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.int2intent = int2intent
        self.intent2int = {intent: index for index, intent in int2intent.items()}
        k = 1.0 / math.sqrt(self.hidden_dim)
        self.u = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim).uniform_(-k, k))
        self.v = nn.Parameter(torch.eye(hidden_dim, input_size))
        self.bias = nn.Parameter(torch.FloatTensor(hidden_dim, 1).uniform_(-k, k))
        self.total_intents = total_intents
        self.fc = nn.Linear(hidden_dim, self.total_intents)

    def forward(self, x):
        hidden = self.init_hidden()
        hidden = self.rnncell(x, hidden)
        hidden = self.fc(hidden.t())
        return hidden

    def init_hidden(self):
        hidden = torch.zeros(self.hidden_dim, 1)
        return hidden

    # RNN Cell used to get output of RNN
    # Partially taken from PyTorch's own RNN cell
    def rnncell(self, sentence, hidden):
        for word in sentence:
            if torch.isinf(word).any():
                return hidden

            igates = torch.mm(self.v, word.view(-1, 1))
            hgates = torch.mm(self.u, hidden)
            hidden = torch.tanh(igates + hgates + self.bias)
        return hidden

    def check_finite(self):
        for name, param in self.named_parameters():
            if torch.isinf(param).any():
                raise Exception("Model Contains inf model parameters.")
            elif torch.isnan(param).any():
                raise Exception("Model Contains nan model parameters.")
