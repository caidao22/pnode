########################################
# Example of usage:
#   In this example, we consider two scenarios for the DAE problem:
#   1. The algebraic constraints are known
##     One can do
#        python3 pendulum_DAE.py --double_prec --implicit_form -ts_trajectory_type memory
#      The best model obtained during the training will be saved to ./train_results/best_pendulum_dae.pth
#      The saved model can be loaded in other runs with --hotstart
#
#   2. The algebraic constraints are unknown
#      Since this scenario is much harder to train because of stability issues, we can leverage
#      the pretrained model. For example, one can do
#        python3 pendulum_DAE.py --double_prec --implicit_form -ts_trajectory_type memory --unknown_alg --pretrained
#
# Tips for training:
#   1. --hotstart allows you to load the best trained model and continue the training fram the saved point.
#      You can use --niters to specify the number of iterations before termination.
#
#   2. The NN that approximates the algebraic constraints is difficult to initialize. The Jacobian for the algebraic
#      constraints affects the stability of the ODE solve. You need a lot of trials to kickstart the training.
#      To tune the initial weights, you can use --init_mean and --init_std
#
#   3. Learning rate is another important hyperparameter to tune. The AdamW optimizer typically requires lr to be small 
#      (1e-2 --1e-6). If the training fails because of diverged nonlinear solve, you can restart it with the options 
#      --pretrained --hotstart --lr <smaller_value>
#######################################
import os
import argparse
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import sys

import copy
import scipy
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=22)
matplotlib.rc('axes', titlesize=22)
matplotlib.use('Agg')

parser = argparse.ArgumentParser('Index-1 Pendulum DAE')
parser.add_argument('--pnode_method', type=str, choices=['beuler', 'cn'], default='cn')
parser.add_argument('--data_size', type=int, default=5)
parser.add_argument('--steps_per_data_point', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--lr', type=float, default=0.01) # should not be smaller than 1e-4
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--activation', type=str, choices=['gelu', 'tanh'], default='gelu')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--implicit_form', action='store_true')
parser.add_argument('--use_dlpack', action='store_true')
parser.add_argument('--double_prec', action='store_true')
parser.add_argument('--train_dir', type=str, metavar='PATH', default='./train_results' )
parser.add_argument('--hotstart', action='store_true')
parser.add_argument('--petsc_ts_adapt', action='store_true')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--unknown_alg', action='store_true')
parser.add_argument('--lr_scheduler', action='store_true')
parser.add_argument('--init_mean', type=float, default=0.0)
parser.add_argument('--init_std', type=float, default=0.1)

args, unknown = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

device = 'cpu'
# data generation
theta_0=np.pi/2
theta_dot0=0.
m=1
r0=1
g=9.81
x0 =  r0*np.sin(theta_0)
y0 = -r0*np.cos(theta_0)
vx0 = r0*theta_dot0*np.cos(theta_0)
vy0 = r0*theta_dot0*np.sin(theta_0)
lbda_0 = (m*r0*theta_dot0**2 +  m*g*np.cos(theta_0))/r0 # equilibrium along the rod's axis
true_y0 = np.array([x0,y0,vx0,vy0,lbda_0])
true_y0 = torch.from_numpy(true_y0).double().to(device)
t = torch.linspace(0., 0.5, args.data_size+1, dtype=torch.float64)

if not args.petsc_ts_adapt:
    unknown.append('-ts_adapt_type')
    unknown.append('none') # disable adaptor in PETSc
    t_traj = torch.linspace(start=0, end=0.5, steps=args.data_size+1+(args.data_size)*(args.steps_per_data_point-1))
    step_size = t_traj[1] - t_traj[0]
else:
    step_size = 1e-2

class Lambda(nn.Module):
    def forward(self, t, y):
        f = torch.clone(y)
        f[0] = y[2]
        f[1] = y[3]
        f[2] = -y[0]*y[4]
        f[3] = -y[1]*y[4] - g
        f[4] = y[4]*(y[0]**2 + y[1]**2) + g*y[1] - (y[2]**2 + y[3]**2)
        return f

M = torch.eye(5)
M[-1,-1] = 0.
M = M.double().to(device)

import petsc4py
sys.argv = [sys.argv[0]] + unknown
petsc4py.init(sys.argv)
from pnode import petsc_adjoint

ode0 = petsc_adjoint.ODEPetsc()
ode0.setupTS(true_y0, Lambda(), step_size=step_size, enable_adjoint=False, use_dlpack=args.use_dlpack, implicit_form=True, method='cn', mass=M)
true_y = ode0.odeint(true_y0, t)

if not args.double_prec:
    true_y = true_y.float()
    t = t.float()
    M = M.float()
true_y = true_y.to(device)
t = t.to(device)
true_y0 = true_y[0]

# delete the last dimension? algebraic variable should not enter data

def get_batch():
    s = torch.from_numpy(np.sort(np.random.choice(np.arange(1, args.data_size+1, dtype=np.int64), args.batch_size, replace=False)))
    s = torch.cat((torch.tensor([0]), s))
    batch_t = t[s]
    batch_y = true_y[s]
    return batch_t, batch_y

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if args.viz:
    makedirs(os.path.join(args.train_dir, 'png'))
    # import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 12), facecolor='white')
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    plt.show(block=False)

#marker_style1 = dict(marker='o', markersize=8, mfc='None')
#marker_style2 = dict(marker='x', markersize=8, mfc='None')
marker_style1 = {}
marker_style2 = {}
lw = 2.5

def visualize(t, true_y, pred_y, odefunc, itr, name):
    if args.viz:
        ax1.cla()
        ax1.set_xlabel('t')
        ax1.set_ylabel(r'$x$')
        ax1.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0], 'g-', linewidth=lw, label='Ground Truth', **marker_style1)
        ax1.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0], 'b--', linewidth=lw, label=name, **marker_style2)
        ax1.legend()

        ax2.cla()
        ax2.set_xlabel('t')
        ax2.set_ylabel(r'$y$')
        ax2.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 1], 'g-', linewidth=lw, **marker_style1)
        ax2.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 1], 'b--', linewidth=lw, **marker_style2)

        ax3.cla()
        ax3.set_xlabel('t')
        ax3.set_ylabel(r'$v_x$')
        ax3.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 2], 'g-', linewidth=lw, **marker_style1)
        ax3.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 2], 'b--', linewidth=lw, **marker_style2)

        ax4.cla()
        ax4.set_xlabel('t')
        ax4.set_ylabel(r'$v_y$')
        ax4.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 3], 'g-', linewidth=lw, **marker_style1)
        ax4.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 3], 'b--', linewidth=lw, **marker_style2)

        ax5.cla()
        ax5.set_xlabel('t')
        ax5.set_ylabel(r'$Lambda$')
        ax5.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 4], 'g-', linewidth=lw, **marker_style1)
        ax5.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 4], 'b--', linewidth=lw, **marker_style2)

        fig.tight_layout()
        plt.savefig(os.path.join(args.train_dir, 'png')+'/{:03d}'.format(itr)+name)
        plt.draw()
        plt.pause(0.001)

# network setup
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        if args.activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Tanh()
        if args.double_prec:
            self.net = nn.Sequential(
                nn.Linear(5, 10, bias=False).double(),
                self.act.double(),
                nn.Linear(10, 10, bias=False).double(),
                self.act.double(),
                nn.Linear(10, 10, bias=False).double(),
                self.act.double(),
                nn.Linear(10, 10, bias=False).double(),
                self.act.double(),
                nn.Linear(10, 4, bias=False).double(),
            ).to(device)
            if args.unknown_alg:
                self.net_alg = nn.Sequential(
                    nn.Linear(5, 10, bias=False).double(),
                    self.act.double(),
                    nn.Linear(10, 10, bias=False).double(),
                    self.act.double(),
                    nn.Linear(10, 1, bias=False).double(),
                ).to(device)
        else:
            self.net = nn.Sequential(
                nn.Linear(5, 10, bias=False),
                self.act,
                nn.Linear(10, 10, bias=False),
                self.act,
                nn.Linear(10, 10, bias=False),
                self.act,
                nn.Linear(10, 10, bias=False),
                self.act,
                nn.Linear(10, 5, bias=False),
            ).to(device)
            if args.unknown_alg:
                self.net_alg = nn.Sequential(
                    nn.Linear(5, 10, bias=False),
                    self.act,
                    nn.Linear(10, 10, bias=False),
                    self.act,
                    nn.Linear(10, 1, bias=False),
                ).to(device)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.001, std=0.01)
        if args.unknown_alg:
            for m in self.net_alg.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=args.init_mean, std=args.init_std)
                    # nn.init.eye_(m.weight)
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        f = torch.clone(y)
        f[:-1] = self.net(y)
        if args.unknown_alg:
            f[-1] = self.net_alg(y)
        else:
            f[-1] = y[4]*(y[0]**2 + y[1]**2) + g*y[1] - (y[2]**2 + y[3]**2)
        return f

if __name__ == '__main__':

    ii = 0
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)

    func_PNODE = ODEFunc().to(device)
    if args.pretrained:
        ckpt_path = os.path.join(args.train_dir, 'best_pendulum_dae.pth')
        ckpt = torch.load(ckpt_path,map_location=device)
        func_PNODE.load_state_dict(ckpt['func_state_dict'], strict=False)
        if args.unknown_alg:
            func_PNODE.net.requires_grad_(False)

    ode_PNODE = petsc_adjoint.ODEPetsc()
    ode_PNODE.setupTS(true_y0, func_PNODE, step_size=step_size, method=args.pnode_method, enable_adjoint=True, implicit_form=args.implicit_form, use_dlpack=args.use_dlpack, mass=M)
    if args.pretrained:
        optimizer_PNODE = optim.AdamW(filter(lambda p: p.requires_grad, func_PNODE.parameters()), lr=args.lr)
    else:
        optimizer_PNODE = optim.AdamW(func_PNODE.parameters(), lr=args.lr)
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_PNODE, milestones=[1000, 10000])
    else:
        scheduler = None
    ode_test_PNODE = petsc_adjoint.ODEPetsc()
    ode_test_PNODE.setupTS(true_y0, func_PNODE, step_size=step_size, method=args.pnode_method, enable_adjoint=False, implicit_form=args.implicit_form, use_dlpack=args.use_dlpack, mass=M)

    loss_PNODE_array = []
    curr_iter = 1
    best_loss = float('inf')

    if args.hotstart:
        ckpt_path = os.path.join(args.train_dir, 'best_pendulum_dae.pth' if not args.unknown_alg else 'best_pendulum_dae_unknown_alg.pth')
        ckpt = torch.load(ckpt_path,map_location=device)
        curr_iter = ckpt['iter'] + 1
        ii = ckpt['ii'] + 1
        best_loss = ckpt['best_loss']
        func_PNODE.load_state_dict(ckpt['func_state_dict'])
        optimizer_PNODE.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer_PNODE.param_groups[0]['lr'] = args.lr
        # optimizer_PNODE.param_groups[0]['weight_decay'] = 0

    loss_save = torch.zeros(args.niters)
    deviation_save = torch.zeros(args.niters)

    lam = 0.01
    loss_func = nn.MSELoss()
    start_PNODE = time.time()
    for itr in range(curr_iter, args.niters + 1):
        batch_t, batch_y = get_batch()
        optimizer_PNODE.zero_grad()
        pred_y_PNODE = ode_PNODE.odeint_adjoint(true_y0, batch_t)
        nfe_f_PNODE = func_PNODE.nfe
        func_PNODE.nfe = 0
        loss_PNODE = loss_func(pred_y_PNODE, batch_y)
        # loss_PNODE = torch.mean(torch.abs(pred_y_PNODE - batch_y))
        if args.unknown_alg:
            loss_PNODE = loss_PNODE + lam*torch.sum(torch.square(func_PNODE.net(pred_y_PNODE)[:,-1]))
        loss_PNODE.backward()
        optimizer_PNODE.step()
        if args.lr_scheduler:
            scheduler.step()

        loss_save[itr-1] = loss_PNODE
        deviation_save[itr-1] = torch.sum(torch.square(func_PNODE.net(pred_y_PNODE)[:,-1]))

        nfe_b_PNODE = func_PNODE.nfe
        func_PNODE.nfe = 0

        if itr % args.test_freq == 0:
            with torch.no_grad():
                end_PNODE = time.time()
                test_t = t
                test_y = true_y
                pred_y_PNODE = ode_test_PNODE.odeint_adjoint(true_y0, test_t)
                loss_PNODE_array= loss_PNODE_array + [loss_PNODE.item()]
                print('PNODE: Iter {:05d} | Time {:.6f} | Total Loss {:.6f} | NFE-F {:04d} | NFE-B {:04d}'.format(itr, end_PNODE-start_PNODE, loss_PNODE_array[-1], nfe_f_PNODE, nfe_b_PNODE))
                print('LR: {:g} Weight_Decay: {:g}'.format(optimizer_PNODE.param_groups[0]['lr'],optimizer_PNODE.param_groups[0]['weight_decay']))
                if args.unknown_alg:
                    print('Check algebraic constraints, total squared deviation: {:g}'.format(deviation_save[itr-1].item()))
                if loss_PNODE_array[-1] < best_loss:
                    best_loss = loss_PNODE_array[-1]
                    new_best = True
                else:
                    new_best = False
                if new_best:
                    visualize(test_t, test_y, pred_y_PNODE, func_PNODE, ii, 'PNODE')
                    ckpt_path = os.path.join(args.train_dir, 'best_pendulum_dae.pth' if not args.unknown_alg else 'best_pendulum_dae_unknown_alg.pth')
                    torch.save({
                        'iter': itr,
                        'ii': ii,
                        'best_loss': best_loss,
                        'func_state_dict': func_PNODE.state_dict(),
                        'optimizer_state_dict': optimizer_PNODE.state_dict(),
                    }, ckpt_path)
                    print('Saved new best results (loss={:.6f}) at Iter {}'.format(best_loss,itr))
                ii += 1
                start_PNODE = time.time()

    np.save('loss_save.npy', loss_save.detach().cpu().numpy())
    np.save('deviation_save.npy', deviation_save.detach().cpu().numpy())
