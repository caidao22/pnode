#!/usr/bin/env python3
########################################
# Example of usage:
#   python3 KS.py --double_prec --implicit_form -ts_trajectory_type memory
# IMEX:
#   python3 KS.py -ts_trajectory_type memory --pnode_model imex --petsc_ts_adapt -ts_adapt_type none --batch_size 512
#   python3 KS.py -ts_trajectory_type memory --pnode_model imex --petsc_ts_adapt -ts_adapt_type none --batch_size 512 -pnode_inner_ksp_hpddm_type gmres
# Prerequisites:
#   pnode petsc4py scipy matplotlib torch tensorboardX

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
    default="mlp",
)
parser.add_argument(
    "--pnode_method",
    type=str,
    choices=["euler", "rk2", "fixed_bosh3", "rk4", "fixed_dopri5", "beuler", "cn"],
    default="cn",
)
parser.add_argument("--normalize", type=str, choices=["minmax", "mean"], default=None)
parser.add_argument("--data_size", type=int, default=40)
parser.add_argument("--steps_per_data_point", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=40)
parser.add_argument("--niters", type=int, default=10000)
parser.add_argument("--test_freq", type=int, default=10)
parser.add_argument("--viz", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--adjoint", action="store_true")
parser.add_argument("--implicit_form", action="store_true")
parser.add_argument("--double_prec", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train_dir", type=str, metavar="PATH", default="./train_results")
parser.add_argument("--hotstart", action="store_true")
parser.add_argument("--petsc_ts_adapt", action="store_true")
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--tb_log", action="store_true")
parser.add_argument("--use_hpddm", action="store_true")
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
    step_size = 0.25*4


def get_data(spatial_stride=1, temporal_stride=1):
    train_data_path = "./training_data_N100000.pickle"
    test_data_path = "./testing_data_N100000.pickle"

    with open(train_data_path, "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        data = pickle.load(file)
        input_sequence = data["train_input_sequence"]
        N_train, dim = np.shape(input_sequence)
        # N_train = 1000
        dt = data["dt"]
        initial_state_train = torch.from_numpy(input_sequence[0, spatial_stride // 2 :: spatial_stride])
        u_train = input_sequence[:N_train:temporal_stride, spatial_stride // 2 :: spatial_stride]
        t_train = dt * np.linspace(0, N_train - 1, N_train)[:N_train:temporal_stride]
        del data
    with open(test_data_path, "rb") as file:
        data = pickle.load(file)
        input_sequence = data["test_input_sequence"]
        N_test, dim = np.shape(input_sequence)
        N_test = 2500
        dt = data["dt"]
        initial_state_test = torch.from_numpy(input_sequence[0, spatial_stride // 2 :: spatial_stride])
        u_test = input_sequence[:N_test:temporal_stride, spatial_stride // 2 :: spatial_stride]
        t_test = dt * np.linspace(0, N_test - 1, N_test)[:N_test:temporal_stride]
        del data
    trainloader = DataLoader(
        DistFuncDataset(u_train, t_train, args.double_prec),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
    )
    testloader = DataLoader(
        DistFuncDataset(u_train, t_train, args.double_prec),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
    )
    print("Finished loading data")
    return initial_state_train, initial_state_test, trainloader, testloader


class DistFuncDataset(Dataset):
    def __init__(self, u_array, t_array, double_prec=True):
        if double_prec:
            self.u = torch.from_numpy(u_array).double()
            self.t = torch.from_numpy(t_array).double()
        else:
            self.u = torch.from_numpy(u_array.astype(np.float32))
            self.t = torch.from_numpy(t_array.astype(np.float32))

    def __len__(self):
        return len(self.u) - 1

    def __getitem__(self, index):
        a = self.u[index]
        b = self.u[index + 1]
        c = self.t[index]
        d = self.t[index + 1]
        return index, a, b, c, d


def split_and_preprocess(
    u, t, batch_size, sizes=[0.8, 0.2], seed=42, write=False, preprocess=None
):
    ## SPLIT DATA into train/val/test sets
    N_all = u.shape[0]
    inds = np.arange(N_all)

    num_train = int(np.floor(sizes[0] * N_all))
    num_test = int(np.floor(sizes[1] * N_all))
    np.random.seed(seed)
    np.random.shuffle(inds)

    train_inds = inds[:num_train]
    test_inds = inds[num_train:]

    if write:
        fh = h5py.File("preprocessed.h5", "w")

    for name, subinds in zip(["train", "test"], [train_inds, test_inds]):
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
        elif "test" in name:
            testloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=0,
            )

    if write:
        fh.close()
    del u, t
    return trainloader, testloader


initial_state_train, initial_state_test, trainloader, testloader = get_data(
    spatial_stride=8,
    temporal_stride=32,
)  # reduce dof from 512 to 64


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


def visualize(t, true_y, pred_u, odefunc, itr, name):
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
            pred_u.cpu().numpy()[:, 0],
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
            pred_u.cpu().numpy()[:, 1],
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
            pred_u.cpu().numpy()[:, 2],
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
    # petsc4py_path = os.path.join(os.environ['PETSC_DIR'],os.environ['PETSC_ARCH'],'lib')
    # sys.path.append(petsc4py_path)
    import petsc4py

    sys.argv = [sys.argv[0]] + unknown
    petsc4py.init(sys.argv)
    from pnode import petsc_adjoint

    ii = 0
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    if args.tb_log:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.train_dir)

    ode_PNODE = petsc_adjoint.ODEPetsc()
    if args.pnode_model == "mlp":
        from models.mlp import ODEFunc
    if args.pnode_model == "snode":
        from models.snode import ODEFunc
    if args.pnode_model == "imex":
        from models.imex import ODEFuncIM, ODEFuncEX

        if args.double_prec:
            funcIM_PNODE = ODEFuncIM().double().to(device)
            funcEX_PNODE = ODEFuncEX().double().to(device)
        else:
            funcIM_PNODE = ODEFuncIM().to(device)
            funcEX_PNODE = ODEFuncEX().to(device)
        ode_PNODE.setupTS(
            torch.zeros(
                args.batch_size,
                *initial_state_train.shape,
                dtype=torch.float64 if args.double_prec else torch.float32,
                device=device
            ),
            funcIM_PNODE,
            step_size=step_size,
            method="imex",
            enable_adjoint=True,
            implicit_form=args.implicit_form,
            imex_form=True,
            func2=funcEX_PNODE,
            batch_size=args.batch_size,
            use_hpddm=args.use_hpddm,
            matrixfree_hpddm=False,
        )
        params = list(funcIM_PNODE.parameters()) + list(funcEX_PNODE.parameters())
        optimizer_PNODE = optim.AdamW(params, lr=args.lr)
        ode_test_PNODE = petsc_adjoint.ODEPetsc()
        ode_test_PNODE.setupTS(
            torch.zeros(
                args.batch_size,
                *initial_state_train.shape,
                dtype=torch.float64 if args.double_prec else torch.float32,
                device=device
            ),
            funcIM_PNODE,
            step_size=step_size,
            method="imex",
            enable_adjoint=False,
            implicit_form=args.implicit_form,
            imex_form=True,
            func2=funcEX_PNODE,
            batch_size=args.batch_size,
            use_hpddm=args.use_hpddm,
            matrixfree_hpddm=False,
        )
    else:
        if args.double_prec:
            func_PNODE = ODEFunc().double().to(device)
        else:
            func_PNODE = ODEFunc().to(device)
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
        ode_test_PNODE = petsc_adjoint.ODEPetsc()
        ode_test_PNODE.setupTS(
            torch.zeros(
                args.batch_size,
                *initial_state_train.shape,
                dtype=torch.float64 if args.double_prec else torch.float32,
                device=device
            ),
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

    stdoutmode = "w+"
    if args.hotstart:
        stdoutmode = "a+"
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
        if args.pnode_model == "imex":
            funcIM_PNODE.load_state_dict(ckpt["funcIM_state_dict"])
            funcEX_PNODE.load_state_dict(ckpt["funcEX_state_dict"])
        else:
            func_PNODE.load_state_dict(ckpt["func_state_dict"])
        optimizer_PNODE.load_state_dict(ckpt["optimizer_state_dict"])
    stdoutfile = open(args.train_dir+'/stdout.log', stdoutmode)
    sys.stdout = udt.Tee(sys.stdout, stdoutfile)

    optimizer_PNODE.param_groups[0]["lr"] = args.lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_PNODE, patience=2, factor=0.5, min_lr=1e-7
    )
    start_PNODE = time.time()
    # torch.cuda.profiler.cudart().cudaProfilerStart()
    for itr in range(curr_iter, args.niters + 1):
        for inner, (indices, u_data, u_target, t_data, t_target) in enumerate(
            trainloader
        ):
            u_data, u_target = u_data.to(device), u_target.to(device)
            optimizer_PNODE.zero_grad()
            pred_u_PNODE = ode_PNODE.odeint_adjoint(u_data, torch.tensor([0.25*4]))
            loss_PNODE = torch.mean(torch.abs(pred_u_PNODE - u_target))
            loss_std_PNODE = torch.std(torch.abs(pred_u_PNODE - u_target))
            loss_PNODE.backward()
            optimizer_PNODE.step()

            if args.pnode_model == "imex":
                params = list(funcIM_PNODE.parameters()) + list(
                    funcEX_PNODE.parameters()
                )
            else:
                params = func_PNODE.parameters()
            if args.tb_log:
                total_norm = 0
                for p in params:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                    print(p.grad)
                total_norm = total_norm**0.5
                writer.add_scalar("Train/Loss", loss_PNODE.item(), itr * 50000)
                writer.add_scalar("Train/Gradient", total_norm, itr * 50000)

        if itr % args.test_freq == 0:
            end_PNODE = time.time()
            with torch.no_grad():
                ntests = 0
                for inner, (indices, u_data, u_target, t_data, t_target) in enumerate(
                    testloader
                ):
                    u_data, u_target = u_data.to(device), u_target.to(device)
                    ntests += 1
                    pred_u_PNODE = ode_test_PNODE.odeint_adjoint(
                        u_data, torch.tensor([0.25*4])
                    )
                    loss_PNODE_array = (
                        loss_PNODE_array
                        + [loss_PNODE.item()]
                        + [torch.mean(torch.abs(pred_u_PNODE - u_target)).cpu()]
                    )
                    loss_std_PNODE_array = (
                        loss_std_PNODE_array
                        + [loss_std_PNODE.item()]
                        + [torch.std(torch.abs(pred_u_PNODE - u_target)).cpu()]
                    )
                avg_test_loss = sum(loss_PNODE_array[-ntests:]) / ntests
                scheduler.step(avg_test_loss)
                print(
                    "PNODE: Iter {:05d} | Train Time {:.3f} | Avg Test Loss {:.6f} | LR {:.3e}".format(
                        itr,
                        end_PNODE - start_PNODE,
                        avg_test_loss,
                        optimizer_PNODE.param_groups[0]["lr"],
                    )
                )
                if avg_test_loss < best_loss:
                    best_loss = avg_test_loss
                    new_best = True
                else:
                    new_best = False
                if new_best:
                    # visualize(test_t, test_y, pred_u_PNODE, func_PNODE, ii, "PNODE")
                    ckpt_path = os.path.join(args.train_dir, "best.pth")
                    if args.pnode_model == "imex":
                        torch.save(
                            {
                                "iter": itr,
                                "ii": ii,
                                "best_loss": best_loss,
                                "funcIM_state_dict": funcIM_PNODE.state_dict(),
                                "funcEX_state_dict": funcEX_PNODE.state_dict(),
                                "optimizer_state_dict": optimizer_PNODE.state_dict(),
                                "normalize_option": args.normalize,
                            },
                            ckpt_path,
                        )
                    else:
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
    # torch.cuda.profiler.cudart().cudaProfilerStop()
    del udt.Tee
    stdoutfile.close()
