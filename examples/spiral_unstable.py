########################################
# Example of usage:
#   python3 spiral_unstable.py -ts_adapt_type none -ts_trajectory_type memory --double_prec --ref_method rk2 --pnode_method cn --niters 200 --test_freq 10 --implicit_form
# Prerequisites:
#   pnode torchvision tensorboardX pytorch_model_summary petsc4py
#######################################
import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.use('Agg')
sys.path.append("../")
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--ref_method', type=str, choices=['euler', 'rk2', 'fixed_bosh3', 'rk4', 'fixed_dopri5'], default='euler')
parser.add_argument('--pnode_method', type=str, choices=['euler', 'rk2', 'fixed_bosh3', 'rk4', 'fixed_dopri5', 'beuler', 'cn'], default='euler')
parser.add_argument('--step_size',type=float, default=.1)#0.1
parser.add_argument('--data_size', type=int, default=161)#161
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--implicit_form', action='store_true')
parser.add_argument('--double_prec', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use_dlpack', action='store_true')
args, unknown = parser.parse_known_args()

# Set these random seeds, so everything can be reproduced.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.double_prec:
    print('Using float64')
    true_y0 = torch.tensor([[2., 0.]], dtype=torch.float64).to(device)
    t = torch.linspace(0., 16., args.data_size, dtype=torch.float64)
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]], dtype=torch.float64).to(device)
else:
    print('Using float32 (PyTorch default)')
    true_y0 = torch.tensor([[2., 0.]] ).to(device)
    t = torch.linspace(0., 16., args.data_size)
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

petsc4py_path = os.path.join(os.environ['PETSC_DIR'],os.environ['PETSC_ARCH'],'lib')
sys.path.append(petsc4py_path)
import petsc4py
sys.argv = [sys.argv[0]] + unknown
petsc4py.init(sys.argv)
from petsc4py import PETSc
from pnode import petsc_adjoint

options = {}
options.update({'step_size':args.step_size})

class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A).to(device)

with torch.no_grad():
    # Generate data
    ode0 = petsc_adjoint.ODEPetsc()
    ode0.setupTS(true_y0, Lambda(), step_size=0.001, method='dopri5_fixed', enable_adjoint=False, use_dlpack=args.use_dlpack)
    #true_y = ode0.odeint_adjoint(true_y0,t)
    true_y = ode0.odeint(true_y0, t)

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if args.viz:
    makedirs('png')
    # import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

def visualize(true_y, pred_y, odefunc, itr, name):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        if args.double_prec:
            dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).double().to(device)).cpu().detach().numpy()
        else:
            dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1) + 1E-6
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr)+name)
        plt.draw()
        plt.pause(0.001)

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()
        if args.double_prec:
            self.fc1 = nn.Linear(2, 2, bias=False).double().to(device)
        else:
            self.fc1 = nn.Linear(2, 2, bias=False).to(device)
        for m in self.fc1.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.1)
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        #return self.net(y**3)
        return self.fc1(y**3)

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

#   Reference
    func_REF = ODEFunc().to(device)
    ode_REF = petsc_adjoint.ODEPetsc()
    ode_REF.setupTS(torch.zeros(args.batch_size,1,2).to(device,true_y0.dtype), func_REF, step_size=args.step_size, method=args.ref_method, enable_adjoint=True)
    optimizer_REF = optim.RMSprop(func_REF.parameters(), lr=1e-2)
#   model for test
    ode_test_REF = petsc_adjoint.ODEPetsc()
    ode_test_REF.setupTS(true_y0.to(device), func_REF, step_size=args.step_size, method=args.ref_method, enable_adjoint=False)
#   end of Reference

#   PNODE implementation
    func_PNODE = copy.deepcopy(func_REF).to(device)
    ode_PNODE = petsc_adjoint.ODEPetsc()
    ode_PNODE.setupTS(torch.zeros(args.batch_size,1,2).to(device,true_y0.dtype), func_PNODE, step_size=args.step_size, method=args.pnode_method, enable_adjoint=True, implicit_form=args.implicit_form)
    optimizer_PNODE = optim.RMSprop(func_PNODE.parameters(), lr=1e-2)
#   model for test
    ode_test_PNODE = petsc_adjoint.ODEPetsc()
    ode_test_PNODE.setupTS(true_y0.to(device), func_PNODE, step_size=args.step_size, method=args.pnode_method, enable_adjoint=False, implicit_form=args.implicit_form)
#   end of PNODE
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    loss_REF_array=[]
    loss_PNODE_array = []
    loss_std_REF_array = []
    loss_std_PNODE_array = []
    dot_product_array = []
    for itr in range(1, args.niters + 1):
        # for p1, p2 in zip(func_REF.parameters(), func_PNODE.parameters()):
        #     p1.data = p2.data.clone()

        optimizer_REF.zero_grad()
        optimizer_PNODE.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()

        start_REF = time.time()
        #pred_y_REF = odeint(func_REF, batch_y0.to(device), batch_t.to(device),method=args.ref_method,options=options).to(device)
        pred_y_REF = ode_REF.odeint_adjoint(batch_y0.to(device), batch_t.to(device))
        loss_REF = torch.mean(torch.abs(pred_y_REF.to(device) - batch_y.to(device)))
        loss_std_REF = torch.std(torch.abs(pred_y_REF.to(device) - batch_y.to(device)))
        end_REF = time.time()
        nfe_f_REF = func_REF.nfe
        func_REF.nfe = 0

        start_PNODE = end_REF
        pred_y_PNODE = ode_PNODE.odeint_adjoint(batch_y0.to(device), batch_t.to(device))
        loss_PNODE = torch.mean(torch.abs(pred_y_PNODE.to(device) - batch_y.to(device)))
        loss_std_PNODE = torch.std(torch.abs(pred_y_PNODE.to(device) - batch_y.to(device)))
        end_PNODE = time.time()
        nfe_f_PNODE = func_PNODE.nfe
        func_PNODE.nfe = 0

        loss_REF.backward()
        optimizer_REF.step()
        nfe_b_REF = func_REF.nfe
        func_REF.nfe = 0

        loss_PNODE.backward()
        optimizer_PNODE.step()
        nfe_b_PNODE = func_PNODE.nfe
        func_PNODE.nfe = 0

        #   inner product between the gradients from two implementations
        num_diff = 0
        norm_diff = 0
        total_num = 0
        array = []
        array2 = []
        for p1, p2 in zip(func_REF.parameters(), func_PNODE.parameters()):
            if np.abs(p1.data.cpu().ne(p2.data.cpu()).sum()) > 1:
                num_diff += 1
                norm_diff += np.abs(p1.data.cpu().ne(p2.data.cpu()).sum())
                array = array + [p1.grad.min().cpu().detach().numpy().tolist()]
                array2 = array2 + [p2.grad.min().cpu().detach().numpy().tolist()]
            total_num += 1

        unit_array = array / (np.linalg.norm(array) + 1E-16)
        unit_array2 = array2 / (np.linalg.norm(array2) + 1E-16)
        dot_product = np.dot(unit_array, unit_array2)

        if itr % args.test_freq == 0:
            with torch.no_grad():
                #pred_y_REF = odeint(func_REF, true_y0.to(device), t.to(device),method=args.ref_method,options=options)
                pred_y_REF = ode_test_REF.odeint_adjoint(true_y0.to(device), t.to(device))
                loss_REF_array=loss_REF_array + [loss_REF.item()]+[torch.mean(torch.abs(pred_y_REF.to(device) - true_y.to(device))).cpu()]
                loss_std_REF_array = loss_std_REF_array + [loss_std_REF.item()]+[torch.std(torch.abs(pred_y_REF.to(device) - true_y.to(device))).cpu()]
                print('REF  : Iter {:04d} | Time {:.6f} | Total Loss {:.6f} | NFE-F {:04d} | NFE-B {:04d}'.format(itr,end_REF-start_REF, loss_REF_array[-1],nfe_f_REF, nfe_b_REF))

                pred_y_PNODE = ode_test_PNODE.odeint_adjoint(true_y0.to(device), t.to(device))
                loss_PNODE_array= loss_PNODE_array + [loss_PNODE.item()]+[torch.mean(torch.abs(pred_y_PNODE.to(device) - true_y.to(device))).cpu()]
                loss_std_PNODE_array = loss_std_PNODE_array + [loss_std_PNODE.item()]+[torch.std(torch.abs(pred_y_PNODE.to(device) - true_y.to(device))).cpu()]
                print('PNODE: Iter {:04d} | Time {:.6f} | Total Loss {:.6f} | NFE-F {:04d} | NFE-B {:04d}'.format(itr,end_PNODE-start_PNODE, loss_PNODE_array[-1],nfe_f_PNODE, nfe_b_PNODE))
                dot_product_array = dot_product_array + [dot_product]
                print('Dot product of normalized gradients: {:.6f} | number of different params: {:04d} / {:04d}\n'.format(dot_product, num_diff, total_num))
                visualize(true_y, pred_y_REF, func_REF, ii, 'REF')
                visualize(true_y, pred_y_PNODE, func_PNODE, ii, 'PNODE')
                ii += 1

        end = time.time()
    print(func_REF.fc1.weight.data)
    print(func_PNODE.fc1.weight.data)

    f = plt.figure(figsize=(10,4))
    ax2 = f.add_subplot(121)
    ax = f.add_subplot(122)
    #ax = plt.figure()
    #plt.figure()
    ax.grid()
    ax2.grid()
    if args.ref_method == 'euler':
        ax.plot(range(0,itr,args.test_freq), [loss_REF_array[2*i+1] for i in range(0, round(itr/args.test_freq))], 'b*-', label='Explicit')
        ax.plot(range(0,itr,args.test_freq), [loss_PNODE_array[2*i+1] for i in range(0, round(itr/args.test_freq))], 'g*-', label='Implicit')
    else:
        ax.plot(range(0,itr,args.test_freq), [loss_REF_array[2*i+1] for i in range(0, round(itr/args.test_freq))], 'b*-', label='Explicit')
        ax.plot(range(0,itr,args.test_freq), [loss_PNODE_array[2*i+1] for i in range(0, round(itr/args.test_freq))], 'g*-', label='Implicit')

    loss_REF_array = np.nan_to_num(np.asarray(loss_REF_array))
    loss_std_REF_array = np.nan_to_num(np.asarray(loss_std_REF_array))
    loss_PNODE_array = np.nan_to_num(np.asarray(loss_PNODE_array))
    loss_std_PNODE_array = np.nan_to_num(np.asarray(loss_std_PNODE_array))
    print(loss_std_PNODE_array)
    print(loss_PNODE_array)

    ax.fill_between(range(0,itr,args.test_freq),[loss_REF_array[2*i+1]-loss_std_REF_array[2*i+1] for i in range(0, round( itr/args.test_freq ) )],
                         [loss_REF_array[2*i+1]+loss_std_REF_array[2*i+1] for i in range(0, round( itr/args.test_freq ) )], alpha=0.1,
                         color="b")
    ax.fill_between(range(0,itr,args.test_freq),[loss_PNODE_array[2*i+1]-loss_std_PNODE_array[2*i+1] for i in range(0, round( itr/args.test_freq ) )],
                         [loss_PNODE_array[2*i+1]+loss_std_PNODE_array[2*i+1] for i in range(0, round( itr/args.test_freq ) )], alpha=0.1,
                         color="g")
    if args.ref_method == 'euler':
        ax2.plot(range(0,itr,args.test_freq), [loss_REF_array[2*i] for i in range(0,round( itr/args.test_freq ))], 'bo-',label='Explicit')
        ax2.plot(range(0,itr,args.test_freq), [loss_PNODE_array[2*i] for i in range(0,round( itr/args.test_freq ))], 'go-',label='Implicit')
    else:
        ax2.plot(range(0,itr,args.test_freq), [loss_REF_array[2*i] for i in range(0,round( itr/args.test_freq ))], 'bo-',label='Explicit')
        ax2.plot(range(0,itr,args.test_freq), [loss_PNODE_array[2*i] for i in range(0,round( itr/args.test_freq ))], 'go-',label='Implicit')

    ax2.fill_between(range(0,itr,args.test_freq),[loss_REF_array[2*i]-loss_std_REF_array[2*i] for i in range(0, round( itr/args.test_freq ) )],
                         [loss_REF_array[2*i]+loss_std_REF_array[2*i] for i in range(0, round( itr/args.test_freq ) )], alpha=0.1,
                         color="b")
    ax2.fill_between(range(0,itr,args.test_freq),[loss_PNODE_array[2*i]-loss_std_PNODE_array[2*i] for i in range(0, round( itr/args.test_freq ) )],
                         [loss_PNODE_array[2*i]+loss_std_PNODE_array[2*i] for i in range(0, round( itr/args.test_freq ) )], alpha=0.1,
                         color="g")
    ax2.set_ylim(-0.05,0.5)
    ax.set_ylim(-0.05,8)
    ax.set_xlabel('Niter')
    ax2.set_xlabel('Niter')
    ax.legend()
    #ax.set_title('NFE-F NODE {:04d}, PNODE {:04d}'.format(nfe_f_REF,nfe_f_PNODE))

    # ax2.plot(dot_product_array,'x',label='Dot product between normalized gradients')
    # ax2.legend()
    # ax2.set_title('Time NODE {:.6f}, PNODE {:.6f}'.format(end_REF-start_REF,end_PNODE-start_PNODE))
    ax.set_title('Test loss',fontweight='bold')
    ax2.set_title('Train loss',fontweight='bold')
    ax2.legend()

    plt.savefig('loss_'+args.ref_method+str(args.seed)+'.png', bbox_inches='tight')
