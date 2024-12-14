#!/usr/bin/env python3
########################################
# This script uses the vanilla neural ODE
# Example of usage:
#   python3 KS_node.py --double_prec --max_epochs 100 --data_size 5000 --batch_size 100 --node_model snode
# Prerequisites:
#   torchdiffeq scipy matplotlib torch tensorboard

#######################################
import os
import time
import torch
import torch.optim as optim
import sys

from KS import *


def get_raw_data():
    with open("./training_data_L22_S64_N50000.pickle", "rb") as file:
        data = pickle.load(file)
        a = data["train_input_sequence"][::10]
        M, trunc = np.shape(a)
        ts = 0.2 * np.linspace(0, M - 1, M)
    train_y = torch.from_numpy(a[: args.data_size, :trunc])
    train_t = torch.from_numpy(ts[: args.data_size])
    val_y = torch.from_numpy(a[args.data_size :, :trunc])
    val_t = torch.tensor(ts[args.data_size :])
    if args.double_prec:
        train_y = train_y.double()
        train_t = train_t.double()
        val_y = val_y.double()
        val_t = val_t.double()
    else:
        train_y = train_y.float()
        train_t = train_t.float()
        val_y = val_y.float()
        val_t = val_t.float()
    return trunc, train_t, train_y, val_t, val_y


# Gets a batch of y from the data evolved forward in time (default 20)
def get_batch(t, true_y):
    length = len(t)
    s = torch.from_numpy(
        np.random.choice(
            np.arange(length - args.time_window_size - 1, dtype=np.int64),
            args.batch_size,
            replace=False,
        )
    )
    u_data = true_y[s]  # (M, D)
    t_data = t[: args.time_window_size + 1]  # (T)
    u_target = torch.stack(
        [true_y[s + i] for i in range(args.time_window_size + 1)], dim=0
    )  # (T, M, D)
    return u_data, t_data, u_target


if __name__ == "__main__":
    (
        train_initial_state,
        val_initial_state,
        trainloader,
        valloader,
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
    # dim, train_t, train_y, val_t, val_y = get_raw_data()
    # dx = 22 / dim

    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    stdoutmode = "a+"
    stdoutfile = open(args.train_dir + "/stdout.log", stdoutmode)
    # sys.stdout = udt.Tee(sys.stdout, stdoutfile)
    print(" ".join(sys.argv))
    if args.tb_log:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(args.train_dir)

    from torchdiffeq import odeint

    if args.pnode_model == "mlp":
        from models.mlp import ODEFunc
    if args.pnode_model == "snode":
        from models.snode import ODEFunc
    if args.double_prec:
        func = (
            ODEFunc(input_size=dim, hidden=dim * 25 // 8, fixed_linear=True, dx=dx)
            .double()
            .to(device)
        )
    else:
        func = ODEFunc(
            input_size=dim, hidden=dim * 25 // 8, fixed_linear=True, dx=dx
        ).to(device)
    optimizer = optim.AdamW(func.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.75, min_lr=1e-5
    )
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    loss_array = []
    loss_std_array = []
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
        curr_epoch = ckpt["iter"] + 1
        ii = ckpt["ii"] + 1
        best_loss = ckpt["best_loss"]
        func.load_state_dict(ckpt["func_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if args.lr != default_lr:  # reset scheduler
        optimizer.param_groups[0]["lr"] = args.lr
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.75, min_lr=1e-5
        )

    loss = torch.nn.MSELoss()
    start = time.time()
    # torch.cuda.profiler.cudart().cudaProfilerStart()
    for itr in range(curr_epoch, args.max_epochs + 1):
        # optimizer.zero_grad()
        # u_data, t_data, u_target = get_batch(train_t, train_y)
        # u_data, u_target = u_data.to(device), u_target.to(device)
        # train_pred_u = odeint(func, u_data, t_data, method="rk4")
        # train_loss = loss(train_pred_u, u_target)
        # train_loss_std = loss(train_pred_u, u_target)
        # train_loss.backward()
        # optimizer.step()

        for inner, (indices, u_data, u_target, t_data, t_target) in enumerate(
            trainloader
        ):
            u_data, u_target = u_data.to(device), u_target.to(device)
            u_target = u_target.movedim(1, 0)
            optimizer.zero_grad()
            train_pred_u = odeint(
                func,
                u_data,
                torch.from_numpy(train_pred_time),
                method=args.node_method,
                options=dict(step_size=args.step_size),
            )
            train_loss = loss(train_pred_u[1:], u_target)
            train_loss_std = torch.std(torch.abs(train_pred_u[1:] - u_target))
            train_loss.backward()
            optimizer.step()

            params = func.parameters()
            if args.tb_log:
                # total_norm = 0
                # for p in params:
                #     param_norm = p.grad.detach().data.norm(2)
                #     total_norm += param_norm.item() ** 2
                #     print(p.grad)
                # total_norm = total_norm**0.5
                writer.add_scalar("Train/Loss", train_loss.item(), itr * 50000)
                # writer.add_scalar("Train/Gradient", total_norm, itr * 50000)

        if itr % args.validate_freq == 0:
            end = time.time()
            with torch.no_grad():
                nvals = 0
                # if True:
                #     u_data, t_data, u_target = get_batch(val_t, val_y)
                #     u_data, u_target = u_data.to(device), u_target.to(device)
                #     val_pred_u = odeint(func, u_data, t_data)
                for inner, (indices, u_data, u_target, t_data, t_target) in enumerate(
                    valloader
                ):
                    u_data, u_target = u_data.to(device), u_target.to(device)
                    u_target = u_target.movedim(1, 0)
                    nvals += 1
                    val_pred_u = odeint(
                        func,
                        u_data,
                        torch.from_numpy(val_pred_time),
                        method=args.node_method,
                        options=dict(step_size=args.step_size),
                    )
                    loss_array = loss_array + [loss(val_pred_u[1:], u_target).cpu()]
                    loss_std_array = loss_std_array + [
                        torch.std(torch.abs(val_pred_u[1:] - u_target)).cpu()
                    ]
                avg_val_loss = sum(loss_array[-nvals:]) / nvals
                scheduler.step(avg_val_loss)
                print(
                    "NODE: Epoch {:05d} | Train Time {:.3f} | Avg Val Loss {:.3e} | LR {:.3e}".format(
                        itr,
                        end - start,
                        avg_val_loss,
                        optimizer.param_groups[0]["lr"],
                    )
                )
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    new_best = True
                else:
                    new_best = False
                if new_best:
                    # visualize(val_t, val_y, train_pred_u, func, ii, "NODE")
                    ckpt_path = os.path.join(
                        args.train_dir,
                        "best_float64.pth" if args.double_prec else "best_float32.pth",
                    )
                    if args.pnode_model == "imex":
                        torch.save(
                            {
                                "epoch": itr,
                                "ii": ii,
                                "best_loss": best_loss,
                                "funcIM_state_dict": funcIM_NODE.state_dict(),
                                "funcEX_state_dict": funcEX_NODE.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "normalize_option": args.normalize,
                            },
                            ckpt_path,
                        )
                    else:
                        torch.save(
                            {
                                "epoch": itr,
                                "ii": ii,
                                "best_loss": best_loss,
                                "func_state_dict": func.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "normalize_option": args.normalize,
                            },
                            ckpt_path,
                        )
                    print(
                        "    Saved new best results (loss={:.3e}) at Epoch {}".format(
                            best_loss, itr
                        )
                    )
                ii += 1
                start = time.time()
    # torch.cuda.profiler.cudart().cudaProfilerStop()
    # del udt.Tee
    stdoutfile.close()
