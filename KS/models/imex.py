import torch
import torch.nn as nn
import math


class ODEFuncIM(nn.Module):
    def __init__(self):
        super(ODEFuncIM, self).__init__()
        self.A = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding='same', padding_mode='circular', bias=False)
        for m in self.A.modules():
            nn.init.uniform_(m.weight, a=-math.sqrt(1.0 / 3.0), b=math.sqrt(1.0 / 3.0))
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        linear_term = self.A(torch.unsqueeze(y, 1))
        linear_term = torch.squeeze(linear_term, 1)
        return linear_term


class ODEFuncEX(nn.Module):
    def __init__(self, input_size=64, hidden=200):
        super(ODEFuncEX, self).__init__()
        self.input_size = input_size
        self.hidden = hidden
        act = nn.ReLU
        self.F = nn.Sequential(
            nn.Linear(self.input_size, self.hidden),
            act(),
            nn.Linear(self.hidden, self.hidden),
            act(),
            nn.Linear(self.hidden, self.hidden),
            act(),
            nn.Linear(self.hidden, self.hidden),
            act(),
            nn.Linear(self.hidden, self.input_size),
        )
        for m in self.F.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.00, std=0.01)
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        return self.F(y)
