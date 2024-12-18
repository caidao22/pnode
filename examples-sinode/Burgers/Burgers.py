#! /home/linot/anaconda3/bin/python3
import sys

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import math

##################
# Example of usage:
#       python3 ODE_Ying.py --adjoint --pnode --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 1bee -snes_type ksponly
#
# Notes:
#   - By default this script use torchdiffeq
#   - To use SINODE, add the options "--pnode --imex"
#   - Add "--double_prec" if PETSc is configured with double precision. It is not needed if PETSc is configured with "--with-precision=single"
#   - Add "--use_dlpack -ts_trajectory_type memory" to save checkpoints to memory instead disk
#   - "--snes_type ksponly" makes PETSc to perform one Newton iteration to solve a linear system with the nonlinear SNES solver
##################

###############################################################################
# Arguments from the submit file
###############################################################################
parser = argparse.ArgumentParser("ODE demo")
# These are the relevant sampling parameters
parser.add_argument("--data_size", type=int, default=100)  # IC from the simulation
parser.add_argument("--dt", type=float, default=0)
parser.add_argument(
    "--batch_time", type=int, default=9
)  # Samples a batch covers (this is 10 snaps in a row in data_size)
parser.add_argument(
    "--batch_size", type=int, default=20
)  # Number of IC to calc gradient with each iteration

parser.add_argument(
    "--method",
    type=str,
    choices=["euler", "rk2", "bosh3", "rk4", "dopri5", "beuler", "cn"],
    default="dopri5",
)  # only useful when using pnode
parser.add_argument("--niters", type=int, default=100)  # Iterations of training
parser.add_argument(
    "--test_freq", type=int, default=1
)  # Frequency for outputting test loss (by epochs not iterations)
parser.add_argument("--viz", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--adjoint", action="store_true")

parser.add_argument("--pnode", action="store_true")
parser.add_argument("--implicit_form", action="store_true")
parser.add_argument("--double_prec", action="store_true")
parser.add_argument("--use_dlpack", action="store_true")
parser.add_argument("--imex", action="store_true")
parser.add_argument(
    "--step_size",
    type=float,
    default=0.05,
    help="fixed step size when using fixed step solvers e.g. rk4",
)
parser.add_argument(
    "--linear_solver", type=str, choices=["petsc", "hpddm", "torch"], default="petsc"
)
parser.add_argument("--epoch", type=int, default=100)  # Epochs of training
parser.add_argument("--tb_log", action="store_true")
parser.add_argument("--train_dir", type=str, metavar="PATH", default="./train_results")

args, unknown = parser.parse_known_args()

# Add 1 to batch_time to include the IC
args.batch_time += 1


###############################################################################
# Classes
###############################################################################


# This is the class that contains the NN that estimates the RHS
class ODEFunc(nn.Module):
    def __init__(self, N):
        super(ODEFunc, self).__init__()
        # Change the NN architecture here
        self.net = nn.Sequential(
            nn.Linear(N, N * 9 // 8),
            nn.ReLU(),
            nn.Linear(N * 9 // 8, N * 9 // 8),
            nn.ReLU(),
            nn.Linear(N * 9 // 8, N * 9 // 8),
            nn.ReLU(),
            nn.Linear(N * 9 // 8, N * 9 // 8),
            nn.ReLU(),
            nn.Linear(N * 9 // 8, N),
        )

        self.lin = nn.Sequential(
            nn.Linear(N, N, bias=False),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        # The following operations are provided since we know the linear term
        for m in self.lin.modules():
            if isinstance(m, nn.Linear):
                # Prescribing the exact form of the linear term since we know it for Burgers - see Linear() function below
                m.weight = nn.Parameter(torch.from_numpy(Linear(N)).float())
                m.weight.requires_grad = False

        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        # This is the evolution with the NN - linear + nonlinear forward RHS
        return self.lin(y) + self.net(y)

    def getNFE(self):
        return self.nfe

    def resetNFE(self):
        self.nfe = 0


# This is the class that contains the NN that estimates the RHS_EX
class ODEFuncEX(nn.Module):
    def __init__(self, N):
        super(ODEFuncEX, self).__init__()
        # Change the NN architecture here
        self.net = nn.Sequential(
            nn.Linear(N, N * 9 // 8),
            nn.ReLU(),
            nn.Linear(N * 9 // 8, N * 9 // 8),
            nn.ReLU(),
            nn.Linear(N * 9 // 8, N * 9 // 8),
            nn.ReLU(),
            nn.Linear(N * 9 // 8, N * 9 // 8),
            nn.ReLU(),
            nn.Linear(N * 9 // 8, N),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        # This is the evolution with the NN - nonlinear forward RHS
        return self.net(y)

    def getNFE(self):
        return self.nfe

    def resetNFE(self):
        self.nfe = 0


# This is the class that contains the NN that estimates the RHS_IM
class ODEFuncIM(nn.Module):
    def __init__(self, fixed_linear=False, dx=0, alpha=8e-4):
        super(ODEFuncIM, self).__init__()
        self.A = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding="same",
            padding_mode="circular",
            bias=False,
        )
        for m in self.A.modules():
            nn.init.uniform_(m.weight, a=-math.sqrt(1.0 / 3.0), b=math.sqrt(1.0 / 3.0))
        if fixed_linear:
            K = torch.tensor(
                [[[alpha / dx**2, -2.0 * alpha / dx**2, alpha / dx**2]]],
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )
            self.A.weight = nn.Parameter(K, requires_grad=False)
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        linear_term = self.A(torch.unsqueeze(y, 1))
        linear_term = torch.squeeze(linear_term, 1)
        return linear_term


# Romit version
class ODEFuncIM_Romit(nn.Module):
    def __init__(self, N):
        super(ODEFuncIM_Romit, self).__init__()
        # Change the NN architecture here

        self.lin = nn.Sequential(
            nn.Linear(N, N, bias=False),
        )
        # The following operations are provided since we know the linear term
        for m in self.lin.modules():
            if isinstance(m, nn.Linear):
                # Prescribing the exact form of the linear term since we know it for Burgers - see Linear() function below
                m.weight = nn.Parameter(torch.from_numpy(Linear(N)).float())
                m.weight.requires_grad = False

    def forward(self, t, y):
        # This is the evolution with the NN - linear forward RHS
        return self.lin(y)


###############################################################################
# Functions
###############################################################################
def Linear(N):
    alpha = 8.0e-4
    dx2 = np.zeros((N, N))
    for i in range(N):
        dx2[i, i] = -2
        if i == 0:
            dx2[i, -1] = 1
            dx2[i, i + 1] = 1
        elif i == N - 1:
            dx2[i, i - 1] = 1
            dx2[i, 0] = 1
        else:
            dx2[i, i - 1] = 1
            dx2[i, i + 1] = 1

    dx = 1 / N
    A = alpha * (1 / dx**2) * dx2

    return A


# Gets a batch of y from the data evolved forward in time (default 20)
def get_batch(dt, ttorch, utorch, args):
    [IC, length, _] = utorch.shape
    batch_size = args.batch_size
    batch_time = args.batch_time

    x = [[j, i] for i in range(length - batch_time) for j in range(IC)]
    lis = [x[i] for i in np.random.choice(len(x), batch_size, replace=False)]

    for i in range(len(lis)):
        if i == 0:
            batch_y0 = utorch[lis[i][0], lis[i][1]][None, :]
            batch_t = (
                ttorch[lis[i][0], lis[i][1] : lis[i][1] + batch_time][None, :]
                - ttorch[lis[i][0], lis[i][1]][None, None]
            )  # this calculation may introduce error when using single precision
            batch_y = torch.stack(
                [utorch[lis[i][0], lis[i][1] + j] for j in range(batch_time)], dim=0
            )[:, None, :]
        else:
            batch_y0 = torch.cat((batch_y0, utorch[lis[i][0], lis[i][1]][None, :]))
            batch_t = torch.cat(
                (
                    batch_t,
                    ttorch[lis[i][0], lis[i][1] : lis[i][1] + batch_time][None, :]
                    - ttorch[lis[i][0], lis[i][1]][None, None],
                )
            )
            batch_y = torch.cat(
                (
                    batch_y,
                    torch.stack(
                        [utorch[lis[i][0], lis[i][1] + j] for j in range(batch_time)],
                        dim=0,
                    )[:, None, :],
                ),
                axis=1,
            )
    batch_t = batch_t.double() * dt  # use double for better accuracy
    if not args.double_prec:
        batch_t = batch_t.float()
    return batch_y0, batch_t, batch_y


if __name__ == "__main__":
    # Check if there are gpus
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    # Save the model
    if args.imex:
        if args.double_prec:
            prefix = "pnode_double_{}_imex".format(args.linear_solver)
        else:
            prefix = "pnode_single_{}_imex".format(args.linear_solver)
    elif args.pnode:
        if args.double_prec:
            prefix = "pnode_double_{}_{}".format(args.linear_solver, args.method)
        else:
            prefix = "pnode_single_{}_{}".format(args.linear_solver, args.method)
    else:
        if args.double_prec:
            prefix = "node_double_{}".format(args.method)
        else:
            prefix = "node_single_{}".format(args.method)

    if args.tb_log:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(args.train_dir)

    ###########################################################################
    # Import Data
    ###########################################################################
    # [u, t] = pickle.load(open("./Data_T5_IC100.p", "rb"))
    [u, t] = pickle.load(open("./Data_T5_IC100_NX1024.p", "rb"))
    dt = 0.1  # must match the time interval in the data
    t = np.arange(51)
    ttorch = torch.from_numpy(t).repeat(100, 1)

    # u = np.float64(u) if args.double_prec else np.float32(u)
    utorch = torch.tensor(u, dtype=torch.float64 if args.double_prec else torch.float32)
    [IC, T, N] = utorch.shape
    utorch_train = utorch[: int(IC * 0.8)]
    utorch_test = utorch[int(IC * 0.8) :]
    iter_per_epoch = int(0.8 * IC) * (T - args.batch_time) // args.batch_size
    print("iterations per epoch: {:d}".format(iter_per_epoch))
    args.niters = args.epoch * iter_per_epoch

    # Determines what solver to use
    if args.pnode:
        # Experienced PETSc users may switch archs by setting the petsc4py path manually
        # petsc4py_path = os.path.join(os.environ["PETSC_DIR"], os.environ["PETSC_ARCH"], "lib")
        # sys.path.append(petsc4py_path)
        import petsc4py

        sys.argv = [sys.argv[0]] + unknown
        print(sys.argv)
        petsc4py.init(sys.argv)
        from pnode import petsc_adjoint

        # construct an ODEPetsc object for training
        ode_train = petsc_adjoint.ODEPetsc()
        # construct an ODEPetsc object for testing
        ode_test = petsc_adjoint.ODEPetsc()
        if args.double_prec:
            pnode_funcIM = ODEFuncIM(True, 1 / N, 8e-4).double().to(device)
            # pnode_funcIM = ODEFuncIM_Romit(N).double().to(device)
            pnode_funcEX = ODEFuncEX(N).double().to(device)
        else:
            pnode_funcIM = ODEFuncIM(True, 1 / N, 8e-4).to(device)
            pnode_funcEX = ODEFuncEX(N).to(device)
        if args.imex:
            ode_train.setupTS(
                torch.zeros(
                    args.batch_size,
                    N,
                    dtype=torch.float64 if args.double_prec else torch.float32,
                    device=device,
                ),
                pnode_funcIM,
                step_size=args.step_size,
                method="imex",
                enable_adjoint=True,
                implicit_form=args.implicit_form,
                imex_form=True,
                func2=pnode_funcEX,
                batch_size=args.batch_size,
                use_dlpack=args.use_dlpack,
                linear_solver=args.linear_solver,
                matrixfree_jacobian=True,
            )
            params = list(pnode_funcEX.parameters()) + list(pnode_funcIM.parameters())
            ode_test.setupTS(
                torch.zeros(
                    args.batch_size,
                    N,
                    dtype=torch.float64 if args.double_prec else torch.float32,
                    device=device,
                ),
                pnode_funcIM,
                step_size=args.step_size,
                method="imex",
                enable_adjoint=False,
                implicit_form=args.implicit_form,
                imex_form=True,
                func2=pnode_funcEX,
                batch_size=args.batch_size,
                use_dlpack=args.use_dlpack,
                linear_solver=args.linear_solver,
                matrixfree_jacobian=True,
            )
        else:
            if args.double_prec:
                func = ODEFunc(N).double().to(device)
            else:
                func = ODEFunc(N).to(device)
            ode_train.setupTS(
                torch.zeros(
                    args.batch_size,
                    N,
                    dtype=torch.float64 if args.double_prec else torch.float32,
                    device=device,
                ),
                func,
                step_size=args.step_size,
                enable_adjoint=True,
                implicit_form=args.implicit_form,
                use_dlpack=args.use_dlpack,
                method=args.method,
                linear_solver=args.linear_solver,
                matrixfree_jacobian=True,
            )
            params = func.parameters()
            ode_test.setupTS(
                torch.zeros(
                    args.batch_size,
                    N,
                    dtype=torch.float64 if args.double_prec else torch.float32,
                    device=device,
                ),
                func,
                step_size=args.step_size,
                enable_adjoint=False,
                implicit_form=args.implicit_form,
                use_dlpack=args.use_dlpack,
                method=args.method,
                linear_solver=args.linear_solver,
                matrixfree_jacobian=True,
            )
    else:
        if args.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:  # This is the default
            from torchdiffeq import odeint

        ###########################################################################
        # Initialize NN for learning the RHS and setup optimization parms
        ###########################################################################
        if args.double_prec:
            func = ODEFunc(N).double().to(device)
        else:
            func = ODEFunc(N).to(device)
        params = func.parameters()
    optimizer = optim.AdamW(params, lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5 * iter_per_epoch, factor=0.5, min_lr=1e-6
    )
    # optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=int(args.niters / 3), gamma=0.1
    # )

    ii = 0

    ###########################################################################
    # Optimization iterations
    ###########################################################################
    ex = 0
    start = time.time()
    fnfe_accumulated = 0
    bnfe_accumulated = 0
    for itr in range(1, args.niters + 1):
        # Get the batch
        batch_y0, batch_t, batch_y = get_batch(dt, ttorch, utorch_train, args)
        batch_y0, batch_y = batch_y0.to(device), batch_y.to(device)
        batch_t = batch_t[0]
        # Initialzie the optimizer
        optimizer.zero_grad()
        # Make a prediction and calculate the loss
        if args.pnode:
            if args.imex:
                pnode_funcEX.resetNFE()
                pred_y = ode_train.odeint_adjoint(batch_y0, batch_t)  # batch_t.float()
                fnfe = pnode_funcEX.getNFE()
                pnode_funcEX.resetNFE()
            else:
                func.resetNFE()
                pred_y = ode_train.odeint_adjoint(batch_y0, batch_t)
                fnfe = func.getNFE()
                func.resetNFE()
        else:
            func.resetNFE()
            pred_y = odeint(
                func,
                batch_y0,
                batch_t,
                method=args.method,
                options=dict(step_size=args.step_size),  # max_iters=args.max_iters
            )
            fnfe = func.getNFE()
            func.resetNFE()

        loss = torch.mean(
            torch.abs(pred_y - batch_y)
        )  # Compute the mean (because this includes the IC it is not as high as it should be)
        if np.isnan(loss.item()) or np.isinf(loss.item()):
            break
        loss.backward()  # Computes the gradient of the loss (w.r.t to the parameters of the network?)
        # Use the optimizer to update the model
        optimizer.step()
        # get backward nfe
        if args.imex:
            bnfe = pnode_funcEX.getNFE()
        else:
            bnfe = func.getNFE()

        scheduler.step(loss)

        if args.tb_log:
            writer.add_scalar("Train/Loss", loss.item(), itr * 50000)
            # total_norm = 0
            # for p in params:
            #     param_norm = p.grad.detach().data.norm(2)
            #     total_norm += param_norm.item() ** 2
            #     print(p.grad)
            # total_norm = total_norm**0.5
            # writer.add_scalar("Train/Gradient", total_norm, itr * 50000)
        fnfe_accumulated = fnfe_accumulated + fnfe
        bnfe_accumulated = bnfe_accumulated + bnfe
        if itr % (args.test_freq * iter_per_epoch) == 0:
            end = time.time()
            with torch.no_grad():
                # Testing loss
                batch_y0, batch_t, batch_y = get_batch(dt, ttorch, utorch_test, args)
                batch_y0, batch_y = batch_y0.to(device), batch_y.to(device)
                batch_t = batch_t[0]
                if args.pnode:
                    pred_y = ode_test.odeint_adjoint(
                        batch_y0, batch_t
                    )  # batch_t.float()
                else:
                    pred_y = odeint(
                        func,
                        batch_y0,
                        batch_t,
                        method=args.method,
                        options=dict(
                            step_size=args.step_size
                        ),  # max_iters=args.max_iters
                    )
                test_loss = torch.mean(torch.abs(pred_y - batch_y))
                if np.isnan(test_loss.item()) or np.isinf(test_loss.item()):
                    break

                print(
                    "Iter {:04d} | Training Loss {:.6f} | Testing Loss {:.6f} | LR {:.3e} | FNFE {:04d} | BNFE {:04d}".format(
                        itr,
                        loss.item(),
                        test_loss.item(),
                        optimizer.param_groups[0]["lr"],
                        fnfe,
                        bnfe,
                    )
                    + "\n"
                )
                # Output epoch and Loss
                myoutput_info = [
                    itr / (args.test_freq * iter_per_epoch),
                    end - start,
                    loss.item(),
                    fnfe,
                    bnfe,
                    test_loss.item(),
                ]
                if args.tb_log:
                    writer.add_scalar(
                        "Test/Loss", test_loss, itr // (args.test_freq * iter_per_epoch)
                    )
                    writer.add_scalar(
                        "NFE",
                        fnfe_accumulated + bnfe_accumulated,
                        itr // (args.test_freq * iter_per_epoch),
                    )
                    writer.add_scalar(
                        "Train time",
                        end - start,
                        itr // (args.test_freq * iter_per_epoch),
                    )
                ii += 1
                start = time.time()
                fnfe_accumulated = 0
                bnfe_accumulated = 0
    # Save the model
    modelname = os.path.join(args.train_dir, prefix + ".pt")
    if args.imex:
        torch.save(pnode_funcEX.state_dict(), modelname)
    elif args.pnode:
        torch.save(func.state_dict(), modelname)
    else:
        torch.save(func.state_dict(), modelname)
    # pickle.dump(func, open('model.p','wb'))
