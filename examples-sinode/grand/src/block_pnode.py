#need to get a PNODE block here

from base_classes import ODEblock
import torch
from utils import get_rw_adj, gcn_norm_fill_val
from pnode import petsc_adjoint

import torch
from torch import nn
from base_classes import ODEFunc

class ODEFuncIM(ODEFunc):
  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncIM, self).__init__(opt, data, device)
    self.A = nn.Identity()
    self.nfe = 0
 
  def forward(self, t, y):
    self.nfe += 1
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    return -alpha*self.A(y)
    

class PNODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t=torch.tensor([0, 1])):
    super(PNODEblock, self).__init__(odefunc, regularization_fns, opt, data, device, t)

    self.aug_dim = 2 if opt['augment'] else 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    if opt['data_norm'] == 'rw':
      edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                                                   fill_value=opt['self_loop_weight'],
                                                                   num_nodes=data.num_nodes,
                                                                   dtype=data.x.dtype)
    else:
      edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
                                           fill_value=opt['self_loop_weight'],
                                           num_nodes=data.num_nodes,
                                           dtype=data.x.dtype)
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    #set up the RHS function and method
    self.final_func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    #construct an ODEPetsc object for training
    self.ode_train = petsc_adjoint.ODEPetsc()
    #construct an ODEPetsc object for testing
    self.ode_test = petsc_adjoint.ODEPetsc()

    self.funcIM = ODEFuncIM(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)

  def forward(self, x):
    t = self.t.type_as(x)
    #torch.zeros(n_b, *self.nhid, device=x.device)
    reg_states = tuple( torch.zeros(x.size(0)).to(x) for i in range(self.nreg) )
    #state = torch.zeros(x.size()[0],2,x.size()[1])
    state = (x,) + reg_states if self.training and self.nreg > 0 else x
    if self.opt['imex']:
      if self.training:
        self.ode_train.setupTS(
            state,
            self.funcIM,
            step_size=self.opt['step_size'],
            method="imex",
            enable_adjoint=True,
            implicit_form=self.opt['implicit_form'],
            imex_form=True,
            func2=self.odefunc,
            batch_size=state.size(dim=0),
            use_dlpack=self.opt['use_dlpack'],
            linear_solver="petsc",
            # matrixfree_solve=False,
        )
        state_dt = self.ode_train.odeint_adjoint(state, t)
      else:
        self.ode_test.setupTS(
            state,
            self.funcIM,
            step_size=self.opt['step_size'],
            method="imex",
            enable_adjoint=False,
            implicit_form=self.opt['implicit_form'],
            imex_form=True,
            func2=self.odefunc,
            batch_size=state.size(dim=0),
            use_dlpack=self.opt['use_dlpack'],
            linear_solver="petsc",
            # matrixfree_solve=False,
        )
        state_dt = self.ode_test.odeint_adjoint(state, t)
    else:
      if self.training:
        self.ode_train.setupTS(
            state,
            self.final_func,
            step_size=self.opt['step_size'],
            implicit_form=self.opt['implicit_form'],
            use_dlpack=self.opt['use_dlpack'],
            method=self.opt['method'],
            enable_adjoint=True
        )
        state_dt = self.ode_train.odeint_adjoint(state, t)
      else:
        self.ode_test.setupTS(
            state,
            self.final_func,
            step_size=self.opt['step_size'],
            implicit_form=self.opt['implicit_form'],
            use_dlpack=self.opt['use_dlpack'],
            method=self.opt['method'],
            enable_adjoint=False
        )
        state_dt = self.ode_test.odeint_adjoint(state, t)

    if self.training and self.nreg > 0:
      z = state_dt[0,1]
      reg_states = tuple( st[1] for st in state_dt[1:] )
      #import pdb;pdb.set_trace()
      return z, reg_states
    else: 
      z = state_dt[1]
      #import pdb;pdb.set_trace()
      return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
