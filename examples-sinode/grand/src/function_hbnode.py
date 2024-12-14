import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
import numpy as np
from data import get_dataset
from utils import MaxNFEException, squareplus
from base_classes import ODEFunc
from HeavyBallNODE.base import *


class net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels, in_channels)
        self.dense2 = nn.Linear(in_channels, out_channels)
        self.dense3 = nn.Linear(out_channels, out_channels)

    def forward(self, h, x):
        #import pdb;pdb.set_trace()
        out = self.dense1(x)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out)
        return out



class HeavyBallNODEFunc(ODEFunc):
  def __init__(self, in_features, out_features, opt, data, device):
    super(HeavyBallNODEFunc, self).__init__(opt, data, device)
    self.net = net(in_features, out_features)
    self.func = HeavyBallNODE(self.net, corr=0, corrf=True)
  
  def forward(self, t, x):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException

    self.nfe += 1
    #import pdb;pdb.set_trace()
    f = self.func(t,x)
    return f





#if __name__ == '__main__':
#  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#  opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10,
#         'attention_norm_idx': 0, 'add_source': False,
#         'alpha_dim': 'sc', 'beta_dim': 'sc', 'max_nfe': 1000, 'mix_features': False
#         }
#  dataset = get_dataset(opt, '../data', False)
#  t = 1
#  func = HeavyBallNODEFunc(dataset.data.num_features, 6, opt, dataset.data, device)
#  out = func(t, dataset.data.x)