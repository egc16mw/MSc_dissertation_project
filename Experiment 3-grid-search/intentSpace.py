from torch import nn
import torch
import math
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


def normalise_alpha(alpha):
    total_sum = [torch.exp(coordinate) for coordinate in alpha]
    total_sum = torch.stack(total_sum).sum()

    new_alpha = torch.stack([torch.exp(coordinate) / total_sum for coordinate in alpha])
    return new_alpha


class IntentSpace(nn.Module):
    def __init__(self, input_size, hidden_dim, B, simplex):
        super(IntentSpace, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.B = B
        self.c = B
        self.simplex = simplex
        k = 1.0 / math.sqrt(self.hidden_dim)
        self.bias = nn.Parameter(torch.FloatTensor(hidden_dim, 1).uniform_(-k, k))
        self.v = nn.Parameter(torch.eye(hidden_dim, input_size))
        self.b_output = nn.Parameter(torch.FloatTensor(1).uniform_(-k, k))
        self.a = nn.Parameter(torch.FloatTensor(hidden_dim, 1).uniform_(-k, k))
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        # Initalise W
        for x in range(B):
            param = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim).uniform_(-k, k))
            name_w = "W" + str(x)
            setattr(self, name_w, param)

        # Initialise alpha based on if using simplex or Euclidean mode
        if simplex:
            K = torch.tensor([4.0])
            for x in range(B):
                alpha = torch.zeros(B)
                for i in range(B):
                    alpha[i] = -K
                alpha[x] = torch.log(1 - (B - 1) * torch.exp(-K))
                alpha = torch.nn.Parameter(alpha)
                name_alpha = "alpha" + str(x)
                setattr(self, name_alpha, alpha)
        else:
            # Euclidean Model
            for x in range(B):
                alpha = torch.zeros(B)
                alpha[x] = 1
                alpha = torch.nn.Parameter(alpha)
                name_alpha = "alpha" + str(x)
                setattr(self, name_alpha, alpha)

    def forward(self, x, alpha_int, detach=False):
        if detach:
            alpha = self.get_alpha(alpha_int).detach()
        else:
            alpha = self.get_alpha(alpha_int)
        u = 0
        hidden = self.init_hidden()
        # Calculate U
        omega_name = "Omega" + str(alpha_int) + str(0)
        if hasattr(self, omega_name):
            for b in range(0, self.B):
                u += alpha[b] * self.get_omega(alpha_int, b) * self.get_weight(b)
        else:
            for b in range(0, self.B):
                u += alpha[b] * self.get_weight(b)

        hidden = self.rnncell(x, hidden, u)
        hidden = self.fc(hidden.t())
        hidden = torch.sigmoid(hidden)
        return hidden

    def add_alpha(self, alpha):
        print("Adding Alpha " + str(self.c))
        alpha = torch.nn.Parameter(alpha)
        name_alpha = "alpha" + str(self.c)
        setattr(self, name_alpha, alpha)
        self.c += 1

    def add_omegas(self, c):
        print("Adding Omega for " + str(c))
        for x in range(self.B):
            param = nn.Parameter(torch.eye(self.hidden_dim, self.hidden_dim))
            name_omega = "Omega" + str(c) + str(x)
            setattr(self, name_omega, param)

    def init_hidden(self):
        hidden = torch.zeros(self.hidden_dim, 1)
        return hidden

    def get_weight(self, i):
        name = "W" + str(i)
        return getattr(self, name)

    def get_alpha(self, i):
        name = "alpha" + str(i)
        alpha = getattr(self, name)
        if self.simplex:
            return normalise_alpha(alpha)
        else:
            return alpha

    def get_omega(self, c, b):
        name_omega = "Omega" + str(c) + str(b)
        return getattr(self, name_omega)

    def rnncell(self, sentence, hidden, u):
        for word in sentence:
            if torch.isinf(word).any():
                return hidden
            igates = torch.mm(self.v, word.view(-1, 1))
            hgates = torch.mm(u, hidden)
            hidden = torch.tanh(igates + hgates + self.bias)
        return hidden

    def is_alpha_frozen(self):
        frozen = not self.alpha0.requires_grad
        for name, param in self.named_parameters():
            if (name.__contains__("alpha")):
                if param.requires_grad == frozen:
                    raise Exception("Some alphas are frozen and some are not")
        return frozen

    def freeze_alpha(self):
        print("Alpha is frozen")
        for name, param in self.named_parameters():
            if (name.__contains__("alpha")):
                param.requires_grad = False
            if (name.__contains__("W")):
                param.requires_grad = True

    def freeze_W(self):
        print("W is frozen")
        for name, param in self.named_parameters():
            if (name.__contains__("alpha")):
                param.requires_grad = True
            if (name.__contains__("W")):
                param.requires_grad = False

    def check_finite(self):
        for name, param in self.named_parameters():
            if torch.isinf(param).any():
                raise Exception("Model Contains inf model parameters.")
            elif torch.isnan(param).any():
                raise Exception("Model Contains nan model parameters.")
