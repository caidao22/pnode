#!/usr/bin/env python3
########################################
# Example of usage:
#   python3 KS.py --double_prec --implicit_form -ts_trajectory_type memory
# IMEX with PETSc linear solver:
#   python3 KS.py --double_prec -ts_trajectory_type memory --pnode_model imex -ts_adapt_type none
# IMEX with HPDDM linear solver:
#   python3 KS.py --double_prec -ts_trajectory_type memory --pnode_model imex -ts_adapt_type none --linear_solver hpddm -pnode_inner_ksp_hpddm_type gmres
# IMEX with direct linear solver:
#   python3 KS.py --double_prec -ts_trajectory_type memory --pnode_model imex -ts_adapt_type none --linear_solver torch
# More advanced settings:
#   python3 KS.py --pnode_model imex -ts_arkimex_type ars122 -ts_trajectory_type memory --max_epochs 5000 --double_prec --time_window_size 4 --lr 1e-3 -ts_adapt_type none -snes_type ksponly -ksp_rtol 1e-9
# Prerequisites:
#   pnode petsc4py scipy matplotlib torch tensorboard

#######################################
import os
import argparse
import time
import numpy as np
import h5py
import pickle
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import copy
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font", size=22)
matplotlib.rc("axes", titlesize=22)
matplotlib.use("Agg")
sys.path.append("../")
import utils.datatools as udt

parser = argparse.ArgumentParser("KS")
parser.add_argument(
    "--pnode_model",
    type=str,
    choices=["mlp", "snode", "imex"],
    default="snode",
)
parser.add_argument(
    "--pnode_method",
    type=str,
    choices=["euler", "rk2", "fixed_bosh3", "rk4", "fixed_dopri5", "beuler", "cn"],
    default="cn",
)
parser.add_argument(
    "--node_model",
    type=str,
    choices=["mlp", "snode"],
    default="snode",
)
parser.add_argument(
    "--node_method",
    type=str,
    choices=[
        "euler",
        "midpoint",
        "fehlberg2",
        "adaptive_heun",
        "heun3",
        "bosh3",
        "rk4",
        "dopri5",
        "dopri8",
        "implicit_admas",
        "fixed_adams",
        "explicit_adams",
        "scipy_solver",
    ],
    default="dopri5",
)
parser.add_argument("--normalize", type=str, choices=["minmax", "mean"], default=None)
parser.add_argument("--step_size", type=float, default=0.2)
parser.add_argument("--data_size", type=int, default=0)
parser.add_argument("--data_temporal_stride", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--time_window_size", type=int, default=1)
parser.add_argument(
    "--time_window_endpoint", action="store_true"
)  # predict only the endpoint of the time window
parser.add_argument("--max_epochs", type=int, default=10000)
parser.add_argument("--validate_freq", type=int, default=1)
parser.add_argument("--viz", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--adjoint", action="store_true")
parser.add_argument("--implicit_form", action="store_true")
parser.add_argument("--double_prec", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train_dir", type=str, metavar="PATH", default="./train_results")
parser.add_argument("--hotstart", action="store_true")
default_lr = 5e-3
parser.add_argument("--lr", type=float, default=default_lr)
parser.add_argument("--tb_log", action="store_true")
parser.add_argument(
    "--linear_solver", type=str, choices=["petsc", "hpddm", "torch"], default="petsc"
)
args, unknown = parser.parse_known_args()

# Set these random seeds, so everything can be reproduced.
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
initial_state = torch.tensor(
    [[1.0, 0.0, 0.0]], dtype=torch.float64 if args.double_prec else torch.float32
)
step_size = args.step_size


def get_data(
    data_size=0,
    spatial_stride=1,
    temporal_stride=1,
    time_window_size=1,
    time_window_endpoint=False,
):
    # train_data_path = "./training_data_L22_S512_N10000.pickle"
    train_data_path = "training_data_L22_S64_N10000.pickle"
    with open(train_data_path, "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        data = pickle.load(file)
        if data_size == 0:
            input_sequence = data["input_sequence"][:]
        else:
            input_sequence = data["input_sequence"][:data_size]
        N, dim = np.shape(input_sequence)
        train_N = int(0.8 * N)
        dt = data["dt"]
        initial_state_train = torch.from_numpy(
            input_sequence[0, spatial_stride // 2 :: spatial_stride]
        )
        train_u = input_sequence[
            :train_N:temporal_stride, spatial_stride // 2 :: spatial_stride
        ]
        train_t = dt * np.linspace(0, train_N - 1, train_N)[:train_N:temporal_stride]
        train_pred_time = (
            dt
            * temporal_stride
            * np.arange(
                0,
                time_window_size + 1,
                1 if not time_window_endpoint else time_window_size,
            )
        )
        train_pred_time = (
            train_pred_time.astype(np.float64)
            if args.double_prec
            else train_pred_time.astype(np.float32)
        )
        print("Training data dimension: ", train_u.shape)
        print("Training prediction time: ", train_pred_time)

        val_N = int(0.2 * N)  # using fewer data
        initial_state_validate = torch.from_numpy(
            input_sequence[0, spatial_stride // 2 :: spatial_stride]
        )
        val_u = input_sequence[
            :val_N:temporal_stride, spatial_stride // 2 :: spatial_stride
        ]
        val_t = dt * np.linspace(0, val_N - 1, val_N)[:val_N:temporal_stride]
        val_pred_time = (
            dt
            * temporal_stride
            * np.arange(
                0,
                time_window_size + 1,
                1 if not time_window_endpoint else time_window_size,
            )
        )
        val_pred_time = (
            val_pred_time.astype(np.float64)
            if args.double_prec
            else val_pred_time.astype(np.float32)
        )
        print("Validation data dimension: ", val_u.shape)
        print("Validation prediction time: ", val_pred_time)
        del data
    trainloader = DataLoader(
        DistFuncDataset(
            train_u,
            train_t,
            args.double_prec,
            time_window_size=time_window_size,
            time_window_endpoint=time_window_endpoint,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
    )
    validateloader = DataLoader(
        DistFuncDataset(
            val_u,
            val_t,
            args.double_prec,
            time_window_size=time_window_size,
            time_window_endpoint=time_window_endpoint,
        ),
        batch_size=len(val_u) - time_window_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
    )
    print("Finished loading data", flush=True)
    dx = 22 / dim
    return (
        initial_state_train,
        initial_state_validate,
        trainloader,
        validateloader,
        train_pred_time,
        val_pred_time,
        dim,
        dx,
    )


class DistFuncDataset(Dataset):
    def __init__(
        self,
        u_array,
        t_array,
        double_prec=True,
        time_window_size=1,
        time_window_endpoint=False,  # use only the end point or all the points in the window
    ):
        if double_prec:
            self.u = torch.from_numpy(u_array).double()
            self.t = torch.from_numpy(t_array).double()
        else:
            self.u = torch.from_numpy(u_array.astype(np.float32))
            self.t = torch.from_numpy(t_array.astype(np.float32))
        self.time_window_size = time_window_size
        if time_window_endpoint:
            self.start_index = time_window_size
        else:
            self.start_index = 1

    def __len__(self):
        return len(self.u) - self.time_window_size

    def __getitem__(self, index):
        a = self.u[index]
        b = self.u[index + self.start_index : index + 1 + self.time_window_size]
        c = self.t[index]
        d = self.t[index + self.start_index : index + 1 + self.time_window_size]
        return index, a, b, c, d


def split_and_preprocess(
    u, t, batch_size, sizes=[0.8, 0.2], seed=42, write=False, preprocess=None
):
    ## SPLIT DATA into train/val/validate sets
    N_all = u.shape[0]
    inds = np.arange(N_all)

    num_train = int(np.floor(sizes[0] * N_all))
    num_validate = int(np.floor(sizes[1] * N_all))
    np.random.seed(seed)
    np.random.shuffle(inds)

    train_inds = inds[:num_train]
    validate_inds = inds[num_train:]

    if write:
        fh = h5py.File("preprocessed.h5", "w")

    for name, subinds in zip(["train", "validate"], [train_inds, validate_inds]):
        usub = u[subinds]
        tsub = t[subinds]

        if write:
            fh.create_dataset("u_" + name, data=usub)
            fh.create_dataset("t_" + name, data=tsub)
        dataset = DistFuncDataset(usub, tsub, args.double_prec)
        if "train" in name:
            trainloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=0,
            )
        elif "validate" in name:
            validateloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=0,
            )

    if write:
        fh.close()
    del u, t
    return trainloader, validateloader


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


def visualize(t, true_y, pred_u, odefunc, epoch, name):
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
            **marker_style1,
        )
        ax1.plot(
            t.cpu().numpy(),
            pred_u.cpu().numpy()[:, 0],
            color="tab:blue",
            linestyle="dashed",
            linewidth=lw,
            label=name,
            **marker_style2,
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
            **marker_style1,
        )
        ax2.plot(
            t.cpu().numpy(),
            pred_u.cpu().numpy()[:, 1],
            color="tab:blue",
            linestyle="dashed",
            linewidth=lw,
            **marker_style2,
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
            **marker_style1,
        )
        ax3.plot(
            t.cpu().numpy(),
            pred_u.cpu().numpy()[:, 2],
            color="tab:blue",
            linestyle="dashed",
            linewidth=lw,
            **marker_style2,
        )
        ax3.set_xscale("log")

        fig.tight_layout()
        plt.savefig(
            os.path.join(args.train_dir, "png") + "/{:03d}".format(epoch) + name
        )
        plt.draw()
        plt.pause(0.001)


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
    (
        initial_state_train,
        initial_state_validate,
        trainloader,
        validateloader,
        train_pred_time,
        val_pred_time,
        dim,
        dx,
    ) = get_data(
        data_size=args.data_size,
        spatial_stride=1,
        temporal_stride=args.data_temporal_stride,
        time_window_size=args.time_window_size,
        time_window_endpoint=args.time_window_endpoint,
    )

    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    stdoutmode = "a+"
    stdoutfile = open(args.train_dir + "/stdout.log", stdoutmode)
    sys.stdout = udt.Tee(sys.stdout, stdoutfile)
    print(" ".join(sys.argv))
    if args.tb_log:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(args.train_dir)

    # petsc4py_path = os.path.join(os.environ['PETSC_DIR'],os.environ['PETSC_ARCH'],'lib')
    # sys.path.append(petsc4py_path)
    import petsc4py

    sys.argv = [sys.argv[0]] + unknown
    petsc4py.init(sys.argv)
    from pnode import petsc_adjoint

    ode_PNODE = petsc_adjoint.ODEPetsc()
    if args.pnode_model == "mlp":
        from models.mlp import ODEFunc
    if args.pnode_model == "snode":
        from models.snode import ODEFunc
    if args.pnode_model == "imex":
        from models.imex import ODEFuncIM, ODEFuncEX

        if args.double_prec:
            funcIM_PNODE = ODEFuncIM(fixed_linear=True, dx=dx).double().to(device)
            funcEX_PNODE = (
                ODEFuncEX(input_size=dim, hidden=dim * 25 // 8).double().to(device)
            )
        else:
            funcIM_PNODE = ODEFuncIM(fixed_linear=True, dx=dx).to(device)
            funcEX_PNODE = ODEFuncEX(input_size=dim, hidden=dim * 25 // 8).to(device)
        ode_PNODE.setupTS(
            torch.zeros(
                args.batch_size,
                *initial_state_train.shape,
                dtype=torch.float64 if args.double_prec else torch.float32,
                device=device,
            ),
            funcIM_PNODE,
            step_size=step_size,
            method="imex",
            enable_adjoint=True,
            implicit_form=args.implicit_form,
            imex_form=True,
            func2=funcEX_PNODE,
            batch_size=args.batch_size,
            linear_solver=args.linear_solver,
            matrixfree_jacobian=True,
            fixed_jacobian_across_solves=True,
        )
        params = list(funcIM_PNODE.parameters()) + list(funcEX_PNODE.parameters())
        optimizer_PNODE = optim.AdamW(params, lr=args.lr)
        val_ode_PNODE = petsc_adjoint.ODEPetsc()
        val_ode_PNODE.setupTS(
            torch.zeros(
                validateloader.batch_size,
                *initial_state_train.shape,
                dtype=torch.float64 if args.double_prec else torch.float32,
                device=device,
            ),
            funcIM_PNODE,
            step_size=step_size,
            method="imex",
            enable_adjoint=False,
            implicit_form=args.implicit_form,
            imex_form=True,
            func2=funcEX_PNODE,
            batch_size=validateloader.batch_size,
            linear_solver=args.linear_solver,
            matrixfree_jacobian=True,
            fixed_jacobian_across_solves=True,
        )
    else:
        if args.double_prec:
            func_PNODE = (
                ODEFunc(input_size=dim, hidden=dim * 25 // 8, fixed_linear=True, dx=dx)
                .double()
                .to(device)
            )
        else:
            func_PNODE = ODEFunc(
                input_size=dim, hidden=dim * 25 // 8, fixed_linear=True, dx=dx
            ).to(device)
        ode_PNODE.setupTS(
            torch.zeros(
                args.batch_size,
                *initial_state_train.shape,
                dtype=torch.float64 if args.double_prec else torch.float32,
                device=device,
            ),
            func_PNODE,
            step_size=step_size,
            method=args.pnode_method,
            enable_adjoint=True,
            implicit_form=args.implicit_form,
        )
        optimizer_PNODE = optim.AdamW(func_PNODE.parameters(), lr=args.lr)
        val_ode_PNODE = petsc_adjoint.ODEPetsc()
        val_ode_PNODE.setupTS(
            torch.zeros(
                validateloader.batch_size,
                *initial_state_train.shape,
                dtype=torch.float64 if args.double_prec else torch.float32,
                device=device,
            ),
            func_PNODE,
            step_size=step_size,
            method=args.pnode_method,
            enable_adjoint=False,
            implicit_form=args.implicit_form,
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_PNODE, patience=50, factor=0.75, min_lr=1e-6
    )
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    loss_PNODE_array = []
    loss_std_PNODE_array = []
    curr_epoch = 1
    best_loss = float("inf")

    ii = 0
    if args.hotstart:
        ckpt_path = os.path.join(
            args.train_dir,
            "best_float64.pth" if args.double_prec else "best_float32.pth",
        )
        ckpt = torch.load(ckpt_path)
        if args.normalize != ckpt["normalize_option"]:
            sys.exit(
                "Normalize option mismatch. Use --normalize {} instead.".format(
                    ckpt["normalize_option"]
                )
            )
        curr_epoch = ckpt["epoch"] + 1
        ii = ckpt["ii"] + 1
        best_loss = ckpt["best_loss"]
        if args.pnode_model == "imex":
            funcIM_PNODE.load_state_dict(ckpt["funcIM_state_dict"])
            funcEX_PNODE.load_state_dict(ckpt["funcEX_state_dict"])
        else:
            func_PNODE.load_state_dict(ckpt["func_state_dict"])
        optimizer_PNODE.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if args.lr != default_lr:  # reset scheduler
        optimizer_PNODE.param_groups[0]["lr"] = args.lr
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_PNODE, patience=50, factor=0.75, min_lr=1e-6
        )
    # loss = torch.nn.L1Loss()
    loss = torch.nn.MSELoss()
    start_PNODE = time.time()
    nfe0 = funcEX_PNODE.nfe if args.pnode_model == "imex" else func_PNODE.nfe
    # torch.cuda.profiler.cudart().cudaProfilerStart()
    for epoch in range(curr_epoch, args.max_epochs + 1):
        for inner, (indices, u_data, u_target, t_data, t_target) in enumerate(
            trainloader
        ):
            u_data, u_target = u_data.to(device), u_target.to(device)
            u_target = u_target.movedim(1, 0)
            optimizer_PNODE.zero_grad()
            pred_u_PNODE = ode_PNODE.odeint_adjoint(
                u_data, torch.from_numpy(train_pred_time)
            )
            # loss_PNODE = torch.mean(torch.abs(pred_u_PNODE[1:] - u_target))
            train_loss_PNODE = loss(pred_u_PNODE[1:], u_target)
            train_loss_std_PNODE = torch.std(torch.abs(pred_u_PNODE[1:] - u_target))
            train_loss_PNODE.backward()
            optimizer_PNODE.step()

            if args.pnode_model == "imex":
                params = list(funcIM_PNODE.parameters()) + list(
                    funcEX_PNODE.parameters()
                )
            else:
                params = func_PNODE.parameters()
            if args.tb_log:
                writer.add_scalar("Train/Loss", train_loss_PNODE.item(), epoch * 50000)
                # total_norm = 0
                # for p in params:
                #     param_norm = p.grad.detach().data.norm(2)
                #     total_norm += param_norm.item() ** 2
                #     print(p.grad)
                # total_norm = total_norm**0.5
                # writer.add_scalar("Train/Gradient", total_norm, epoch * 50000)

        if epoch % args.validate_freq == 0:
            end_PNODE = time.time()
            nfe1 = funcEX_PNODE.nfe if args.pnode_model == "imex" else func_PNODE.nfe
            with torch.no_grad():
                nval = 0
                for inner, (indices, u_data, u_target, t_data, t_target) in enumerate(
                    validateloader
                ):
                    u_data, u_target = u_data.to(device), u_target.to(device)
                    u_target = u_target.movedim(1, 0)
                    nval += 1
                    pred_u_PNODE = val_ode_PNODE.odeint_adjoint(
                        u_data, torch.from_numpy(val_pred_time)
                    )
                    loss_PNODE_array = loss_PNODE_array + [
                        loss(pred_u_PNODE[1:], u_target).cpu()
                    ]
                    loss_std_PNODE_array = loss_std_PNODE_array + [
                        torch.std(torch.abs(pred_u_PNODE[1:] - u_target)).cpu()
                    ]
                avg_val_loss = sum(loss_PNODE_array[-nval:]) / nval
                scheduler.step(avg_val_loss)
                print(
                    "PNODE: Epoch {:05d} | Train Time {:.3f} | Avg Val Loss {:.3e} | LR {:.3e}".format(
                        epoch,
                        end_PNODE - start_PNODE,
                        avg_val_loss,
                        optimizer_PNODE.param_groups[0]["lr"],
                    )
                )
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    new_best = True
                else:
                    new_best = False
                if new_best:
                    # visualize(validate_t, validate_y, pred_u_PNODE, func_PNODE, ii, "PNODE")
                    ckpt_path = os.path.join(
                        args.train_dir,
                        "best_float64.pth" if args.double_prec else "best_float32.pth",
                    )
                    if args.pnode_model == "imex":
                        torch.save(
                            {
                                "epoch": epoch,
                                "ii": ii,
                                "best_loss": best_loss,
                                "funcIM_state_dict": funcIM_PNODE.state_dict(),
                                "funcEX_state_dict": funcEX_PNODE.state_dict(),
                                "optimizer_state_dict": optimizer_PNODE.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "normalize_option": args.normalize,
                            },
                            ckpt_path,
                        )
                    else:
                        torch.save(
                            {
                                "epoch": epoch,
                                "ii": ii,
                                "best_loss": best_loss,
                                "func_state_dict": func_PNODE.state_dict(),
                                "optimizer_state_dict": optimizer_PNODE.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "normalize_option": args.normalize,
                            },
                            ckpt_path,
                        )
                    print(
                        "    Saved new best results (loss={:.3e}) at Epoch {}".format(
                            best_loss, epoch
                        )
                    )
                if args.tb_log:
                    writer.add_scalar("Test/Loss", avg_val_loss, epoch)
                    writer.add_scalar("NFE", nfe1 - nfe0, epoch)
                    writer.add_scalar("Train time", end_PNODE - start_PNODE, epoch)
                ii += 1
                start_PNODE = time.time()
                nfe0 = (
                    funcEX_PNODE.nfe if args.pnode_model == "imex" else func_PNODE.nfe
                )
    # torch.cuda.profiler.cudart().cudaProfilerStop()
    del udt.Tee
    stdoutfile.close()
