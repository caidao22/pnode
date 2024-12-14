import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import sys

# uncomment the following to make the run deterministic
# torch.manual_seed(0)
# np.random.seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

#
# Exmaple usage:
#  python3 ode_demo_petsc.py --double_prec -ts_adapt_type none -ts_type rk -ts_rk_type 4 -ts_trajectory_type memory -ts_trajectory_solution_only 0
#
# Note:
#   - PETSc4py must be installed. It can be installed with PETSc using the configuration option --with-petsc4py
#   - Must add -ts_adapt_type none to disable adaptive time integration for this example
#   - Add --double_prec if PETSc is configured with double precision. It is not needed if PETSc is configured with --with-precision=single
#   - By default, disk is used for checkpointing. To use DRAM, add -ts_trajectory_type memory. -ts_trajectory_solution_only 0 can be used to further reduce recomputation at the cost of more memory usage

parser = argparse.ArgumentParser("ODE demo")
parser.add_argument(
    "--method", type=str, choices=["euler", "rk2", "bosh3", "rk4", "dopri5", "beuler", "cn"], default="dopri5"
) # must add the option --implicit_form if you want to use implicit methods such as beuler and cn
parser.add_argument("--data_size", type=int, default=1001)
parser.add_argument("--batch_time", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--niters", type=int, default=2000)
parser.add_argument("--test_freq", type=int, default=20)
parser.add_argument("--viz", action="store_true")
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--step_size", type=float, default=0.025)
parser.add_argument("--implicit_form", action="store_true")
parser.add_argument("--double_prec", action="store_true")
parser.add_argument("--use_dlpack", action="store_true")
args, unknown = parser.parse_known_args()

method = args.method
gpu = args.gpu
niters = args.niters
test_freq = args.test_freq
data_size = args.data_size
batch_time = args.batch_time
batch_size = args.batch_size
step_size = args.step_size
implicit_form = args.implicit_form
double_prec = args.double_prec
use_dlpack = args.use_dlpack

# Experienced PETSc users may switch archs by setting the petsc4py path manually
# petsc4py_path = os.path.join(os.environ["PETSC_DIR"], os.environ["PETSC_ARCH"], "lib")
# sys.path.append(petsc4py_path)
import petsc4py

sys.argv = [sys.argv[0]] + unknown
petsc4py.init(sys.argv)
from petsc4py import PETSc

# OptDB = PETSc.Options()
# print("first init: ",OptDB.getAll())

sys.path.append("../")  # for quick debugging
from pnode import petsc_adjoint

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = torch.device('cpu')
if double_prec:
    print("Using float64")
    true_y0 = torch.tensor([[2.0, 0.0]], dtype=torch.float64).to(device)
    t = torch.linspace(0.0, 25.0, data_size, dtype=torch.float64)
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]], dtype=torch.float64).to(device)
else:
    print("Using float32 (PyTorch default)")
    true_y0 = torch.tensor([[2.0, 0.0]]).to(device)
    t = torch.linspace(0.0, 25.0, data_size)
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y**3, true_A)


# data_size-1 should not exceed the number of time steps
if step_size > 25.0 / (data_size - 1):
    print(
        "Error: step_size={} too large (number of steps should not be smaller than data_size={} too large".format(
            step_size, data_size
        )
    )
    # sys.exit()

ode0 = petsc_adjoint.ODEPetsc()
ode0.setupTS(
    true_y0,
    Lambda(),
    step_size=step_size,
    method=method,
    enable_adjoint=False,
    implicit_form=implicit_form,
    use_dlpack=use_dlpack,
)
with torch.no_grad():
    true_y = ode0.odeint(true_y0, t)
    # print(true_y)
    # sys.exit()


def get_batch():
    s = torch.from_numpy(
        np.random.choice(
            np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False
        )
    )
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack(
        [true_y[s + i] for i in range(batch_time)], dim=0
    )  # (T, M, D)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs("png")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor="white")
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title("Trajectories")
        ax_traj.set_xlabel("t")
        ax_traj.set_ylabel("x,y")
        ax_traj.plot(
            t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], "g-"
        )
        ax_traj.plot(
            t.numpy(),
            pred_y.numpy()[:, 0, 0],
            "--",
            t.numpy(),
            pred_y.numpy()[:, 0, 1],
            "b--",
        )
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title("Phase Portrait")
        ax_phase.set_xlabel("x")
        ax_phase.set_ylabel("y")
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], "g-")
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], "b--")
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title("Learned Vector Field")
        ax_vecfield.set_xlabel("x")
        ax_vecfield.set_ylabel("y")

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = (
            odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)))
            .cpu()
            .detach()
            .numpy()
        )
        mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
        dydt = dydt / mag
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig("png/{:03d}".format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        if double_prec:
            self.net = nn.Sequential(
                nn.Linear(2, 50).double(),
                nn.Tanh().double(),
                nn.Linear(50, 2).double(),
            ).to(device)
        else:
            self.net = nn.Sequential(
                nn.Linear(2, 50),
                nn.Tanh(),
                nn.Linear(50, 2),
            ).to(device)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)


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


if __name__ == "__main__":
    ii = 0
    batch_y0, _, _ = get_batch()
    func = ODEFunc()
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    ode = petsc_adjoint.ODEPetsc()
    ode.setupTS(
        batch_y0,
        func,
        step_size=step_size,
        method=method,
        implicit_form=implicit_form,
        use_dlpack=use_dlpack,
    )
    for itr in range(1, niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = ode.odeint_adjoint(batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % test_freq == 0:
            with torch.no_grad():
                ode0.setupTS(
                    true_y0,
                    func,
                    step_size=step_size,
                    method=method,
                    enable_adjoint=False,
                    implicit_form=implicit_form,
                    use_dlpack=use_dlpack,
                )
                pred_y = ode0.odeint_adjoint(true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print("Iter {:04d} | Total Loss {:.6f}".format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()
