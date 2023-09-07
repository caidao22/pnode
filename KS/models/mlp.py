import torch.nn as nn


class ODEFunc(nn.Module):
    def __init__(self, input_size=64, hidden=104):
        super(ODEFunc, self).__init__()
        self.input_size = input_size
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden),
            nn.Sigmoid(),
            nn.Linear(self.hidden, self.hidden),
            nn.Sigmoid(),
            nn.Linear(self.hidden, self.hidden),
            nn.Sigmoid(),
            nn.Linear(self.hidden, self.hidden),
            nn.Sigmoid(),
            nn.Linear(self.hidden, self.input_size),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.00, std=0.01)
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        # return self.net(y**3)
        return self.net(y)
