#!/usr/bin/env python3
import os
import numpy as np
from scipy import integrate
solve_ivp = integrate.solve_ivp
import torch
import torch.nn as nn
import sys
import pytest
import petsc4py

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
initial_state = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
endtime = 1.1e-3 # slightly large than 1e-3, otherwise scipy ode solver may not reach the last evaluation point
t = torch.cat((torch.tensor([0],dtype=torch.float64),torch.logspace(start=-5,end=-3,steps=3,dtype=torch.float64)))
step_size = (t[1:] - t[:-1]).tolist()

petsc_args = []
petsc_args.append('-ts_adapt_type')
petsc_args.append('none') # disable adaptor in PETSc
petsc_args.append('-ts_trajectory_type')
petsc_args.append('memory')
petsc_args.append('-ts_monitor')
sys.argv = sys.argv + petsc_args
petsc4py.init(sys.argv)
from pnode import petsc_adjoint

def fun(t, state):
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    f1 = -k1*state[0] + k3*state[1]*state[2]
    f2 = k1*state[0] - k3*state[1]*state[2] - k2*state[1]**2
    f3 = k2*state[1]**2
    return np.array([f1, f2, f3])

def jac(t, state):
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    return np.array([[-k1, k3*state[2], k3*state[1]],
                    [k1, -2.0*k2*state[1]-k3*state[2], -k3*state[1]],
                    [0, 2.0*k2*state[1], 0]])

def get_data(initial_state, **kwargs):
    if not 'rtol' in kwargs.keys():
        kwargs['rtol'] = 1e-11
        kwargs['atol'] = 1e-14
    t_eval = t.detach().numpy()
    path = solve_ivp(fun=fun, jac=jac, t_span=[0, endtime], y0=initial_state.cpu().flatten(),
                     t_eval=t_eval, **kwargs)
    data = torch.from_numpy(path['y'].T)
    return data

true_y = get_data(initial_state, method='BDF')
true_y = true_y.to(device)
t = t.to(device)
true_y0 = true_y[0]

class Lambda(nn.Module):
    def forward(self, t, y):
        k1 = 0.04
        k2 = 3e7
        k3 = 1e4
        f1 = -k1*y[0] + k3*y[1]*y[2]
        f2 = k1*y[0] - k3*y[1]*y[2] - k2*y[1]**2
        f3 = k2*y[1]**2
        return torch.stack((f1, f2, f3), -1)

def test_petsc_scalartype():
    from petsc4py import PETSc
    assert PETSc.ScalarType == np.float64

def test_petsc_odesolver():
    func_validate = Lambda().to(device)
    ode_validate = petsc_adjoint.ODEPetsc()
    ode_validate.setupTS(true_y0, func_validate, step_size=step_size, method='cn', enable_adjoint=False, implicit_form=True)
    pred_y_validate = ode_validate.odeint_adjoint(true_y0, t)
    loss = torch.mean(torch.abs(pred_y_validate - true_y)).cpu()
    loss_std = torch.std(torch.abs(pred_y_validate - true_y)).cpu()
    print('Loss {:g} | Loss std {:g}'.format(loss,loss_std))
    assert loss.item() == pytest.approx(8.45e-8,abs=1e-8)
    assert loss_std.item() == pytest.approx(1.95e-7,abs=1e-8)

if __name__ == '__main__':
    test_petsc_odesolver()
