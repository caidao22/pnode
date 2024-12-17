#!/usr/bin/env python3
########################################
# Example of usage:
#   python3 ROBER.py --double_prec --implicit_form -ts_trajectory_type memory --normalize minmax
# Note:
#   - Use --double_prec if you have installed PETSc with double precision
# Prerequisites:
#   pnode petsc4py scipy matplotlib torch tensorboardX

#######################################
import os
import argparse
import time
import numpy as np
from scipy import integrate

solve_ivp = integrate.solve_ivp
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import copy
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font", size=22)
matplotlib.rc("axes", titlesize=22)
matplotlib.use("Agg")
sys.path.append("../")
parser = argparse.ArgumentParser("ODE demo")
parser.add_argument(
    "--pnode_method",
    type=str,
    choices=["euler", "rk2", "bosh3", "rk4", "dopri5", "beuler", "cn"],
    default="cn",
)
parser.add_argument("--normalize", type=str, choices=["minmax", "mean"], default=None)
parser.add_argument("--data_size", type=int, default=40)
parser.add_argument("--steps_per_data_point", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=40)
parser.add_argument("--niters", type=int, default=10000)
parser.add_argument("--test_freq", type=int, default=100)
parser.add_argument("--viz", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--adjoint", action="store_true")
parser.add_argument("--implicit_form", action="store_true")
parser.add_argument("--double_prec", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train_dir", type=str, metavar="PATH", default="./train_results")
parser.add_argument("--hotstart", action="store_true")
parser.add_argument("--petsc_ts_adapt", action="store_true")
args, unknown = parser.parse_known_args()

# Set these random seeds, so everything can be reproduced.
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
initial_state = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
endtime = 100.0
# t = torch.linspace(0, endtime, args.data_size, dtype=torch.float64)
t = torch.cat(
    (torch.tensor([0]), torch.logspace(start=-5, end=2, steps=args.data_size))
)
if not args.petsc_ts_adapt:
    unknown.append("-ts_adapt_type")
    unknown.append("none")  # disable adaptor in PETSc
    t_traj = torch.cat(
        (
            torch.tensor([0]),
            torch.logspace(
                start=-5,
                end=2,
                steps=args.data_size
                + (args.data_size - 1) * (args.steps_per_data_point - 1),
            ),
        )
    )
    step_size = (t_traj[1:] - t_traj[:-1]).tolist()
else:
    step_size = 1e-5


def fun(t, state):
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    f1 = -k1 * state[0] + k3 * state[1] * state[2]
    f2 = k1 * state[0] - k3 * state[1] * state[2] - k2 * state[1] ** 2
    f3 = k2 * state[1] ** 2
    return np.array([f1, f2, f3])


def jac(t, state):
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    return np.array(
        [
            [-k1, k3 * state[2], k3 * state[1]],
            [k1, -2.0 * k2 * state[1] - k3 * state[2], -k3 * state[1]],
            [0, 2.0 * k2 * state[1], 0],
        ]
    )


def get_data(initial_state, **kwargs):
    if "rtol" not in kwargs.keys():
        kwargs["rtol"] = 1e-11
        kwargs["atol"] = 1e-14
    t_eval = t.detach().numpy()
    path = solve_ivp(
        fun=fun,
        jac=jac,
        t_span=[0, endtime],
        y0=initial_state.cpu().flatten(),
        t_eval=t_eval,
        **kwargs
    )
    data = torch.from_numpy(path["y"].T)
    if args.normalize == "minmax":
        shift = data.min(0, keepdim=True)[0]
        scale = data.max(0, keepdim=True)[0] - data.min(0, keepdim=True)[0]
        data = (data - shift) / scale
    elif args.normalize == "mean":
        shift = data.mean(0, keepdim=True)
        scale = data.max(0, keepdim=True)[0] - data.min(0, keepdim=True)[0]
        data = (data - shift) / scale
    else:
        shift = torch.zeros_like(data[0])
        scale = torch.ones_like(data[0])
    return data, shift, scale


# reshape(args.data_size, 3)

true_y, shift, scale = get_data(initial_state, method="BDF")
if not args.double_prec:
    true_y = true_y.float()
    t = t.float()
true_y = true_y.to(device)
shift = shift.to(device)
scale = scale.to(device)
t = t.to(device)
true_y0 = true_y[0]


class Lambda(nn.Module):
    def forward(self, t, y):
        k1 = 0.04
        k2 = 3e7
        k3 = 1e4
        f1 = -k1 * y[0] + k3 * y[1] * y[2]
        f2 = k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] ** 2
        f3 = k2 * y[1] ** 2
        return torch.stack((f1, f2, f3), -1)


def get_batch():
    s = torch.from_numpy(
        np.sort(
            np.random.choice(
                np.arange(1, args.data_size + 1, dtype=np.int64),
                args.batch_size,
                replace=False,
            )
        )
    )
    s = torch.cat((torch.tensor([0]), s))
    batch_t = t[s]
    batch_y = true_y[s]
    return batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs(os.path.join(args.train_dir, "png"))
    # import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 12), facecolor="white")
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    plt.show(block=False)

# marker_style1 = dict(marker='o', markersize=8, mfc='None')
# marker_style2 = dict(marker='x', markersize=8, mfc='None')
marker_style1 = {}
marker_style2 = {}
lw = 2.5


def visualize(t, true_y, pred_y, odefunc, itr, name):
    if args.viz:
        ax1.cla()
        ax1.set_xlabel("t")
        ax1.set_ylabel(r"$u_1$")
        ax1.plot(
            t.cpu().numpy(),
            true_y.cpu().numpy()[:, 0],
            color="tab:orange",
            linestyle="solid",
            linewidth=lw,
            label="Ground Truth",
            **marker_style1
        )
        ax1.plot(
            t.cpu().numpy(),
            pred_y.cpu().numpy()[:, 0],
            color="tab:blue",
            linestyle="dashed",
            linewidth=lw,
            label=name,
            **marker_style2
        )
        ax1.legend()
        ax1.set_xscale("log")

        ax2.cla()
        # ax2.set_title('Phase Portrait')
        ax2.set_xlabel("t")
        ax2.set_ylabel(r"$u_2$")
        ax2.plot(
            t.cpu().numpy(),
            true_y.cpu().numpy()[:, 1],
            color="tab:orange",
            linestyle="solid",
            linewidth=lw,
            **marker_style1
        )
        ax2.plot(
            t.cpu().numpy(),
            pred_y.cpu().numpy()[:, 1],
            color="tab:blue",
            linestyle="dashed",
            linewidth=lw,
            **marker_style2
        )
        ax2.set_xscale("log")

        ax3.cla()
        ax3.set_xlabel("t")
        ax3.set_ylabel(r"$u_3$")
        ax3.plot(
            t.cpu().numpy(),
            true_y.cpu().numpy()[:, 2],
            color="tab:orange",
            linestyle="solid",
            linewidth=lw,
            **marker_style1
        )
        ax3.plot(
            t.cpu().numpy(),
            pred_y.cpu().numpy()[:, 2],
            color="tab:blue",
            linestyle="dashed",
            linewidth=lw,
            **marker_style2
        )
        ax3.set_xscale("log")

        fig.tight_layout()
        plt.savefig(os.path.join(args.train_dir, "png") + "/{:03d}".format(itr) + name)
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        if args.double_prec:
            self.net = nn.Sequential(
                nn.Linear(3, 5, bias=False).double(),
                nn.GELU().double(),
                nn.Linear(5, 5, bias=False).double(),
                nn.GELU().double(),
                nn.Linear(5, 5, bias=False).double(),
                nn.GELU().double(),
                nn.Linear(5, 5, bias=False).double(),
                nn.GELU().double(),
                nn.Linear(5, 5, bias=False).double(),
                nn.GELU().double(),
                nn.Linear(5, 5, bias=False).double(),
                nn.GELU().double(),
                nn.Linear(5, 3, bias=False).double(),
            ).to(device)
        else:
            self.net = nn.Sequential(
                nn.Linear(3, 5, bias=False),
                nn.GELU(),
                nn.Linear(5, 5, bias=False),
                nn.GELU(),
                nn.Linear(5, 5, bias=False),
                nn.GELU(),
                nn.Linear(5, 5, bias=False),
                nn.GELU(),
                nn.Linear(5, 5, bias=False),
                nn.GELU(),
                nn.Linear(5, 5, bias=False),
                nn.GELU(),
                nn.Linear(5, 3, bias=False),
            ).to(device)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.5)
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        # return self.net(y**3)
        return self.net(y)


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


def validate_data():
    func_validate = Lambda().to(device)
    ode_validate = petsc_adjoint.ODEPetsc()
    ode_validate.setupTS(
        true_y0,
        func_validate,
        step_size=step_size,
        method=args.pnode_method,
        enable_adjoint=False,
        implicit_form=args.implicit_form,
    )
    pred_y_validate = ode_validate.odeint_adjoint(true_y0, t)
    loss_validate = torch.mean(torch.abs(pred_y_validate - true_y)).cpu()
    loss_std_validate = torch.std(torch.abs(pred_y_validate - true_y)).cpu()
    print(
        "Validate | Loss {:.6f} | Loss std {:.6f}".format(
            loss_validate, loss_std_validate
        )
    )
    visualize(t, true_y, pred_y_validate, func_validate, 0, "validate")


if __name__ == "__main__":
    # Experienced PETSc users may switch archs by setting the petsc4py path manually
    # petsc4py_path = os.path.join(os.environ['PETSC_DIR'],os.environ['PETSC_ARCH'],'lib')
    # sys.path.append(petsc4py_path)
    import petsc4py

    sys.argv = [sys.argv[0]] + unknown
    petsc4py.init(sys.argv)
    from pnode import petsc_adjoint

    ii = 0
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    writer = SummaryWriter(args.train_dir)

    func_PNODE = ODEFunc().to(device)
    ode_PNODE = petsc_adjoint.ODEPetsc()
    ode_PNODE.setupTS(
        true_y0,
        func_PNODE,
        step_size=step_size,
        method=args.pnode_method,
        enable_adjoint=True,
        implicit_form=args.implicit_form,
    )
    optimizer_PNODE = optim.AdamW(func_PNODE.parameters(), lr=5e-3)
    ode_test_PNODE = petsc_adjoint.ODEPetsc()
    ode_test_PNODE.setupTS(
        true_y0,
        func_PNODE,
        step_size=step_size,
        method=args.pnode_method,
        enable_adjoint=False,
        implicit_form=args.implicit_form,
    )

    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    loss_PNODE_array = []
    loss_std_PNODE_array = []
    curr_iter = 1
    best_loss = float("inf")

    if args.hotstart:
        ckpt_path = os.path.join(args.train_dir, "best.pth")
        ckpt = torch.load(ckpt_path)
        if args.normalize != ckpt["normalize_option"]:
            sys.exit(
                "Normalize option mismatch. Use --normalize {} instead.".format(
                    ckpt["normalize_option"]
                )
            )
        curr_iter = ckpt["iter"] + 1
        ii = ckpt["ii"] + 1
        best_loss = ckpt["best_loss"]
        func_PNODE.load_state_dict(ckpt["func_state_dict"])
        optimizer_PNODE.load_state_dict(ckpt["optimizer_state_dict"])

    start_PNODE = time.time()
    for itr in range(curr_iter, args.niters + 1):
        batch_t, batch_y = get_batch()
        optimizer_PNODE.zero_grad()
        pred_y_PNODE = ode_PNODE.odeint_adjoint(true_y0, batch_t)
        nfe_f_PNODE = func_PNODE.nfe
        func_PNODE.nfe = 0
        loss_PNODE = torch.mean(torch.abs(pred_y_PNODE - batch_y))
        loss_std_PNODE = torch.std(torch.abs(pred_y_PNODE - batch_y))
        loss_PNODE.backward()
        optimizer_PNODE.step()
        nfe_b_PNODE = func_PNODE.nfe
        func_PNODE.nfe = 0

        total_norm = 0
        for p in func_PNODE.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        writer.add_scalar("Train/Loss", loss_PNODE.item(), itr * 50000)
        writer.add_scalar("Train/Gradient", total_norm, itr * 50000)

        if itr % args.test_freq == 0:
            with torch.no_grad():
                end_PNODE = time.time()
                # test_t_index = torch.from_numpy(np.arange(0,args.data_size,args.data_size//20, dtype=np.int64))
                # test_t = t[test_t_index]
                # test_y = true_y[test_t_index]
                test_t = t
                test_y = true_y
                pred_y_PNODE = ode_test_PNODE.odeint_adjoint(true_y0, test_t)
                loss_PNODE_array = (
                    loss_PNODE_array
                    + [loss_PNODE.item()]
                    + [torch.mean(torch.abs(pred_y_PNODE - test_y)).cpu()]
                )
                loss_std_PNODE_array = (
                    loss_std_PNODE_array
                    + [loss_std_PNODE.item()]
                    + [torch.std(torch.abs(pred_y_PNODE - test_y)).cpu()]
                )
                print(
                    "PNODE: Iter {:05d} | Time {:.6f} | Total Loss {:.6f} | NFE-F {:04d} | NFE-B {:04d}".format(
                        itr,
                        end_PNODE - start_PNODE,
                        loss_PNODE_array[-1],
                        nfe_f_PNODE,
                        nfe_b_PNODE,
                    )
                )
                # dot_product_array = dot_product_array + [dot_product]
                # print('Dot product of normalized gradients: {:.6f} | number of different params: {:04d} / {:04d}\n'.format(dot_product, num_diff, total_num))
                if args.normalize is not None:
                    test_y = test_y * scale + shift
                    pred_y_PNODE = pred_y_PNODE * scale + shift
                if loss_PNODE_array[-1] < best_loss:
                    best_loss = loss_PNODE_array[-1]
                    new_best = True
                else:
                    new_best = False
                if new_best:
                    visualize(test_t, test_y, pred_y_PNODE, func_PNODE, ii, "PNODE")
                    ckpt_path = os.path.join(args.train_dir, "best.pth")
                    torch.save(
                        {
                            "iter": itr,
                            "ii": ii,
                            "best_loss": best_loss,
                            "func_state_dict": func_PNODE.state_dict(),
                            "optimizer_state_dict": optimizer_PNODE.state_dict(),
                            "normalize_option": args.normalize,
                        },
                        ckpt_path,
                    )
                    print(
                        "Saved new best results (loss={}) at Iter {}".format(
                            best_loss, itr
                        )
                    )
                ii += 1
                start_PNODE = time.time()
    # print(func_PNODE.net[0].weight.data)
