import matplotlib.pyplot as plt
import numpy as np
import pickle

data_path = "./training_data_N100000.pickle"

with open(data_path, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["train_input_sequence"]
    N, dim = np.shape(u)
    dt = data["dt"]
    del data

for N_plot in [1000, 10000, 100000]:
    u_plot = u[:N_plot,:]
    # Plotting the contour plot
    fig = plt.subplots()
    # t, s = np.meshgrid(np.arange(dim)*dt, np.array(range(N))+1)
    n, s = np.meshgrid(np.arange(N_plot), np.array(range(dim))+1)
    plt.contourf(n, s, np.transpose(u_plot), 15, cmap=plt.get_cmap("seismic"))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$n, \quad t=n \cdot {:}$".format(dt))
    plt.savefig("./Figures/Plot_U_first_N{:d}.png".format(N_plot), bbox_inches="tight")
    plt.close()

for N_plot in [1000, 10000, 100000]:
    u_plot = u[-N_plot:,:]
    # Plotting the contour plot
    fig = plt.subplots()
    # t, s = np.meshgrid(np.arange(N_plot)*dt, np.array(range(N))+1)
    n, s = np.meshgrid(np.arange(N_plot), np.array(range(dim))+1)
    plt.contourf(n, s, np.transpose(u_plot), 15, cmap=plt.get_cmap("seismic"))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$n, \quad t=n \cdot {:}$".format(dt))
    plt.savefig("./Figures/Plot_U_last_N{:d}.png".format(N_plot), bbox_inches="tight")
    plt.close()
