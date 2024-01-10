import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import torch

data_path = "training_data_L22_S512_N10000.pickle"
temporal_stride = 10
spatial_stride = 1
L = 22
# step_size = 0.001
step_size = 0.2

with open(data_path, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["input_sequence"][::temporal_stride, spatial_stride // 2 :: spatial_stride]
    u = u[0:]
    initial_state = torch.from_numpy(u[0])
    N, dim = np.shape(u)
    dt = data["dt"] * temporal_stride
    t_pred = dt * np.linspace(0, 2 * N - 1, 2 * N)[:]
    print(t_pred)
    del data


def plot_first(u, name, points, figpath="./"):
    for N_plot in points:
        u_plot = u[:N_plot, :]
        # Plotting the contour plot
        fig = plt.subplots()
        # t, s = np.meshgrid(np.arange(dim)*dt, np.array(range(N))+1)
        n, s = np.meshgrid(
            np.arange(N_plot) * dt, -L / 2 + L / dim * np.array(range(dim))
        )
        plt.contourf(n, s, np.transpose(u_plot), 15, cmap=plt.get_cmap("seismic"))
        plt.colorbar()
        plt.xlabel(r"$t$")
        plt.ylabel(r"$x$")
        plt.savefig(
            "{}/{}_first_N{:d}.png".format(figpath, name, N_plot), bbox_inches="tight"
        )
        plt.close()


def plot_last(u, name, points, figpath="./"):
    for N_plot in points:
        u_plot = u[-N_plot:, :]
        # Plotting the contour plot
        fig = plt.subplots()
        # t, s = np.meshgrid(np.arange(N_plot)*dt, np.array(range(N))+1)
        n, s = np.meshgrid(np.arange(N_plot), np.array(range(dim)) + 1)
        plt.contourf(n, s, np.transpose(u_plot), 15, cmap=plt.get_cmap("seismic"))
        plt.colorbar()
        plt.xlabel(r"$t$")
        plt.ylabel(r"$x$")
        plt.savefig(
            "{}/{}_last_N{:d}.png".format(figpath, name, N_plot), bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    modelpath = "./train_results/best_float64.pth"
    figpath = "./"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelpath", action="store", dest="modelpath", default=modelpath, type=str
    )
    parser.add_argument(
        "--figpath", action="store", dest="figpath", default=figpath, type=str
    )
    parser.add_argument(
        "--pnode_model",
        type=str,
        choices=["mlp", "snode", "imex"],
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
    parser.add_argument("--implicit_form", action="store_true")
    parser.add_argument("--double_prec", action="store_true")
    parser.add_argument(
        "--linear_solver",
        type=str,
        choices=["petsc", "hpddm", "torch"],
        default="petsc",
    )
    args, unknown = parser.parse_known_args()

    if os.path.exists(args.modelpath):
        modelpath = args.modelpath
    else:
        raise FileNotFoundError(
            "Cannot find model file, make sure --modelpath is correct!"
        )
    figpath = args.figpath
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    if args.node_model is not None:
        from torchdiffeq import odeint
    else:
        import petsc4py

        sys.argv = [sys.argv[0]] + unknown
        petsc4py.init(sys.argv)
        from pnode import petsc_adjoint

        ode_PNODE = petsc_adjoint.ODEPetsc()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial_state = initial_state.to(
        device, dtype=torch.float64 if args.double_prec else torch.float32
    )
    if args.pnode_model == "mlp" or args.node_model == "mlp":
        from models.mlp import ODEFunc
    if args.pnode_model == "snode" or args.node_model == "snode":
        from models.snode import ODEFunc
    if args.pnode_model == "imex":
        from models.imex import ODEFuncIM, ODEFuncEX

        if args.double_prec:
            funcIM_PNODE = ODEFuncIM(fixed_linear=True, dx=L / dim).double().to(device)
            funcEX_PNODE = (
                ODEFuncEX(input_size=dim, hidden=dim * 25 // 8).double().to(device)
            )
        else:
            funcIM_PNODE = ODEFuncIM(fixed_linear=True, dx=L / dim).to(device)
            funcEX_PNODE = ODEFuncEX(input_size=dim, hidden=dim * 25 // 8).to(device)
        ode_PNODE.setupTS(
            torch.zeros(
                1,
                *initial_state.shape,
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
            batch_size=1,
            linear_solver=args.linear_solver,
            matrixfree_jacobian=True,
        )
    else:
        if args.double_prec:
            func = (
                ODEFunc(
                    input_size=dim, hidden=dim * 25 // 8, fixed_linear=True, dx=L / dim
                )
                .double()
                .to(device)
            )
        else:
            func = ODEFunc(
                input_size=dim, hidden=dim * 25 // 8, fixed_linear=True, dx=L / dim
            ).to(device)
        if args.node_model is None:
            ode_PNODE.setupTS(
                torch.zeros(
                    1,
                    *initial_state.shape,
                    dtype=torch.float64 if args.double_prec else torch.float32,
                    device=device,
                ),
                func,
                step_size=step_size,
                method=args.pnode_method,
                enable_adjoint=False,
                implicit_form=args.implicit_form,
                linear_solver=args.linear_solver,
                matrixfree_jacobian=True,
            )

    state = torch.load(modelpath, map_location=device)
    if args.pnode_model == "imex":
        funcIM_PNODE.load_state_dict(state["funcIM_state_dict"])
        funcEX_PNODE.load_state_dict(state["funcEX_state_dict"])
    else:
        func.load_state_dict(state["func_state_dict"])

    t_pred = (
        t_pred.astype(np.float64) if args.double_prec else t_pred.astype(np.float32)
    )
    with torch.no_grad():
        if args.node_model is not None:
            pred_u = odeint(
                func,
                initial_state[None],
                torch.from_numpy(t_pred),
                method=args.node_method,
                options=dict(step_size=step_size),
            )
            print(pred_u.shape)
        else:
            pred_u = ode_PNODE.odeint_adjoint(initial_state, torch.from_numpy(t_pred))

    plot_first(pred_u.squeeze(1).cpu().numpy(), "ml", [250, 500, 750], figpath=figpath)
    # plot_last(pred_u.numpy(), "ml", figpath=figpath)
    plot_first(u, "gt", [250, 500, 750], figpath=figpath)
    # plot_last(u, "gt", figpath=figpath)
