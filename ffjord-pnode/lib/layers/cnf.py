import torch
import torch.nn as nn
import os
import sys

# Specify the arch of PETSc being used and initialize PETSc and petsc4py. For this driver, PETSc should be built with single precision.
petsc4py_path = os.path.join(os.environ['PETSC_DIR'],os.environ['PETSC_ARCH'],'lib')
sys.path.append(petsc4py_path)
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Import PNODE
# sys.path.append("../") # for quick debugging
from pnode import petsc_adjoint


from .wrappers.cnf_regularization import RegularizedODEfunc

__all__ = ["CNF"]


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None, solver='dopri5_fixed'):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver
        
        self.test_solver = solver
        
        self.solver_options = {}
        self.init_train = True
        self.init_test = True
        
        self.init_train = False
        self.init_test = False
        self.ode = petsc_adjoint.ODEPetsc()

    def forward(self, z, logpz=None, integration_times=None, reverse=False):

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        odefunc = self.odefunc
        if reverse:
            print('Flipping funcion for integrating backward')
            odefunc = FlipFunc(self.odefunc)
            # re-initialize the TS objects since the ODE function is changed
            self.init_train = False
            self.init_test = False


        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))

        if self.training:
            if self.init_train == False:
                self.ode.setupTS(_flatten((z, _logpz) + reg_states), FlattenFunc(odefunc,(z, _logpz) + reg_states), step_size=self.solver_options.get('step_size'), method=self.solver, enable_adjoint=True)
                self.init_train = True
            state_t = self.ode.odeint_adjoint(_flatten((z, _logpz) + reg_states), integration_times  )
            state_t = _revert_to_tuple(state_t,(z, _logpz) + reg_states)
                #print('train: ', state_t)
        else:
            if self.init_test == False:
                self.ode.setupTS(_flatten((z, _logpz) ), FlattenFunc(odefunc,(z, _logpz) ), step_size=self.solver_options.get('step_size'), method=self.test_solver, enable_adjoint=False)
                self.init_test = True
            state_t = self.ode.odeint_adjoint(_flatten((z, _logpz)), integration_times  )
            state_t = _revert_to_tuple(state_t,(z, _logpz))
                #print('test: ', state_t)

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        self.regularization_states = state_t[2:]


        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def _revert_to_tuple(x, x0):
    out = ()
    idx = 0
    if len(x.shape) == 1:
        
        for x0_ in x0:
            shape=x0_.shape
            out = out + (x[idx:idx +torch.numel(x0_) ].view(*x0_.shape), )
            idx = idx + torch.numel(x0_)
    else:
        xdim = len(x.shape)
        
        for x0_ in x0:
            shape=x0_.shape
            out = out + (x[:,idx:idx +torch.numel(x0_) ].view(xdim,*x0_.shape), )
            idx = idx + torch.numel(x0_)
        
    
    return out

def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

class FlattenFunc(nn.Module):

            def __init__(self, base_func,y0):
                super(FlattenFunc, self).__init__()
                self.base_func = base_func
                self.y0 = y0

            def forward(self, t, y):
                return _flatten( self.base_func(t, _revert_to_tuple(y,self.y0) ) )

class FlipFunc(nn.Module):

    def __init__(self, base_func):
        super(FlipFunc, self).__init__()
        self.base_func = base_func
        self.before_odeint = base_func.before_odeint
        
    def forward(self,t,y):
        return (-f_ for f_ in self.base_func(1-t,y))
        # For integrating backward in time
        


