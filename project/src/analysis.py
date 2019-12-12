#!/usr/bin/env python3

from math import pi
import numpy as np

from matplotlib.style import use
# use("dark_background")
use(["seaborn", "seaborn-paper"])
from matplotlib.cm import get_cmap
from pltconfig import *
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib.animation import FuncAnimation

from initialize import problem_2D

def get_params():

    # discretization parameters
    dt = 5e-1  # time discretization. Keep this number low
    N = 60  # inverse space discretization. Keep this number high!
    M = 60  # inverse space discretization. Keep this number high!

    psi1 = 0.0  # force on system by chemical bath B1
    psi2 = 0.0  # force on system by chemical bath B2

    steady = True # evolve to steady-state distribution

    # modes for solver:
    # 1 := forward-euler
    # 2 := imex
    # 3 := rk2
    # 4 := rk4
    # 5 := rk4 - integrating factor
    # 6 := rk4 - exponential time differencing
    # 7 := gl04
    # 8 := gl06
    # 9 := gl08
    # 10 := gl10

    scheme = 1

    return ( dt, N, M, psi1, psi2, steady, scheme )

def plot_equilibrium():

    [ dt, N, M, psi0, psi1, steady, scheme ] = get_params()

    target_dir = './master_output_dir/'

    problem = problem_2D(
        x0=0.0, xn=2.0*pi, y0=0.0, ym=2.0*pi, n=N, m=M,
        E0=2.0, Ec=8.0, E1=2.0, num_minima0=3.0, num_minima1=3.0,
        D=0.001, psi0=psi0, psi1=psi1
    )

    xx, yy = np.meshgrid(problem.theta0, problem.theta1, indexing="ij")

    fig, ax = subplots(1,1)
    ax.set_xlabel(r"$\theta_{0}$", fontsize=38)
    ax.set_ylabel(r"$\theta_{1}$", fontsize=38, labelpad=12)
    ax.tick_params(labelsize=34)

    temp_fig, temp_ax = subplots(1,1)
    cs = temp_ax.contourf(
        xx, yy,
        np.linspace(
            0.0, 1.05*problem.p_equil.max(),
            problem.n*problem.m
            ).reshape(problem.n, problem.m), 30,
        vmin=0.0, vmax=problem.p_equil.max(), cmap=get_cmap("viridis")
        )

    ax.contourf(
        problem.theta0, problem.theta1, problem.p_equil, 30,
        vmin=0.0, vmax=problem.p_equil.max(), cmap=get_cmap("viridis")
        )

    fig.tight_layout()

    left = 0.12
    right = 0.80
    bottom = 0.1
    top = 0.95

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    cax = fig.add_axes([0.82, 0.10, 0.02, top-bottom])
    cbar = fig.colorbar(cs, cax=cax, orientation="vertical", ax=ax)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0,0))
    cbar.ax.tick_params(labelsize=34)
    cbar.ax.yaxis.offsetText.set_fontsize(28)
    cbar.ax.yaxis.offsetText.set_x(5.75)
    cbar.update_ticks()

    fig.text(0.93, 0.5*(top+bottom), r"$\pi(\bm{\theta})$", fontsize=38, rotation="vertical")

    fig.savefig(target_dir + f"/equilibrium_N_{N}_M_{M}_psi0_{psi0}_psi1_{psi1}_figure.pdf")

def analyze_oversight():

    [ _, _, _, psi0, psi1, _, scheme ] = get_params()

    target_dir = './master_output_dir/'
    data_dir = "/Users/jlucero/data_dir/2019-12-02/"

    dtlist = [5e-1, 1e-1, 1e-2, 1e-3]

    spatial_type = "fd"

    Ndicts = {}; tdicts = {}; errdicts = {}; maxdicts = {
        N: problem_2D(
        x0=0.0, xn=2.0*pi, y0=0.0, ym=2.0*pi, n=N, m=N,
        E0=2.0, Ec=8.0, E1=2.0, num_minima0=3.0, num_minima1=3.0,
        D=0.001, psi0=psi0, psi1=psi1
        ).p_equil.max() for N in [60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    }

    for dt in dtlist:

        data_filename = f'/{spatial_type}_oversight_scheme_{scheme}_dt_{dt}_file.dat'

        try:
            data_array  = np.loadtxt(data_dir + data_filename)

            if not (len(data_array.shape) == 1):
                data_array = data_array[data_array[:,0].argsort()]
                Ndicts[dt] = data_array[:,0]
                tdicts[dt] = data_array[:,3]
                errdicts[dt] = data_array[:,4]
            else:
                Ndicts[dt] = np.array([data_array[0]])
                tdicts[dt] = np.array([data_array[3]])
                errdicts[dt] = np.array([data_array[4]])

        except OSError:
            print(f"{dt} is missing. Moving to next")

    fig, ax = subplots(1,1)
    ax.set_ylabel(r"$t_{\mathrm{convergence}}$", fontsize=38)
    ax.set_xlabel(r"$N$", fontsize=38)

    for ii, dt in enumerate(dtlist):

        try:
            ax.plot(
                Ndicts[dt], tdicts[dt], "o-", 
                color=f"C{ii}", ms=12, label=rf"${dt}$"
                )
        except KeyError:
            print(f"{dt} not present. Not plotting as result.")

    ax.set_xticks([60.0, 120.0, 180.0, 240.0, 300.0, 360.0])
    ax.set_xlim([60.0, 361.0])
    ax.set_ylim([1e-3, 1e3])
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(loc=0, prop={"size": 22}, title=r"$k$", title_fontsize=28)
    ax.tick_params(labelsize=34)

    fig.tight_layout()
    fig.subplots_adjust(top=0.97)
    fig.savefig(target_dir + f"/{spatial_type}_scheme_{scheme}_time_figure.pdf")

    fig2, ax2 = subplots(1,1)
    ax2.set_ylabel(r"$E(N)$", fontsize=38)
    ax2.set_xlabel(r"$N$", fontsize=38)

    for ii, dt in enumerate(dtlist):

        try:
            maxref = np.array([maxdicts[n] for n in Ndicts[dt]])
            ax2.plot(
                Ndicts[dt], errdicts[dt]/maxref, "o-", 
                color=f"C{ii}", ms=12, label=rf"${dt}$"
                )
        except KeyError:
            print(f"{dt} not present. Not plotting as result.")

    ax2.set_xticks([60.0, 120.0, 180.0, 240.0, 300.0, 360.0])
    ax2.set_xlim([60.0, 361.0])
    ax2.set_ylim([1e-15, 1e-1])
    ax2.set_yscale("log")
    ax2.grid(True)
    ax2.legend(loc=0, prop={"size": 22}, title=r"$k$", title_fontsize=24)
    ax2.tick_params(labelsize=34)

    fig2.tight_layout()
    fig2.subplots_adjust(top=0.97)
    fig2.savefig(target_dir + f"/{spatial_type}_scheme_{scheme}_err_figure.pdf")

    fig3, ax3 = subplots(1,1)
    ax3.set_ylabel(r"$t_{\mathrm{convergence}}$", fontsize=38)
    ax3.set_xlabel(r"$E(N)$", fontsize=38)

    for ii, dt in enumerate(dtlist):

        try:
            maxref = np.array([maxdicts[n] for n in Ndicts[dt]])
            ax3.plot(
                errdicts[dt]/maxref, tdicts[dt], "o-",
                color=f"C{ii}", ms=12, label=rf"${dt}$"
                )
        except KeyError:
            print(f"{dt} not present. Not plotting as result.")

    ax3.set_ylim([1e-3, 1e3])
    ax3.set_xlim([1e-15, 1e-1])
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.grid(True)
    ax3.legend(loc=0, prop={"size": 22}, title=r"$k$", title_fontsize=24)
    ax3.tick_params(labelsize=34)

    fig3.tight_layout()
    fig3.subplots_adjust(top=0.97)
    fig3.savefig(
        target_dir + f"/{spatial_type}_scheme_{scheme}_time_err_figure.pdf"
        )

def trace():

    [ dt, N, M, psi0, psi1, steady, scheme ] = get_params()

    target_dir = './master_output_dir/'
    data_filename = f'/evol_scheme_{scheme}_dt_{dt}_N_{N}_M_{M}_psi0_{psi0}_psi1_{psi1}_outfile.dat'
    data_total_path = target_dir + data_filename

    problem = problem_2D(
        x0=0.0, xn=2.0*pi, y0=0.0, ym=2.0*pi, n=N, m=M,
        E0=2.0, Ec=8.0, E1=2.0, num_minima0=3.0, num_minima1=3.0,
        D=0.001, psi0=psi0, psi1=psi1
    )

    xx, yy = np.meshgrid(problem.theta0, problem.theta1, indexing="ij")

    all_data = np.loadtxt(data_total_path)

    data_max = all_data.__abs__().max()

    reverse_purples = get_cmap("Purples_r")

    fig = figure()

    temp_fig, temp_ax = subplots(1,1)
    cs = temp_ax.contourf(
        xx, yy,
        np.linspace(
            0.0, 1.05*all_data.__abs__().max(),
            problem.n*problem.m
            ).reshape(problem.n, problem.m), 30,
        vmin=0.0, vmax=data_max, cmap=reverse_purples
        )

    ax = fig.add_subplot(1,1,1)
    ax.set_ylabel(r"$\theta_{1}$", fontsize=28)
    ax.set_xlabel(r"$\theta_{0}$", fontsize=28)
    ax.set_xlim([problem.x0, problem.xn])
    ax.set_ylim([problem.y0, problem.ym])


    fig.tight_layout()

    left = 0.1
    right = 0.85
    bottom = 0.1
    top = 0.925

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    cax = fig.add_axes([0.88,0.10, 0.02, 0.825])
    cbar = fig.colorbar(cs, cax=cax, orientation="vertical", ax=ax)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0,0))
    cbar.ax.tick_params(labelsize=24)
    cbar.ax.yaxis.offsetText.set_fontsize(24)
    cbar.ax.yaxis.offsetText.set_x(2.75)
    cbar.update_ticks()

    cset = ax.contourf(
            xx, yy, all_data[:,0].reshape(problem.n, problem.m).T,
            30, vmin=0.0, vmax=data_max, cmap=reverse_purples
            )

    def update(frame_number, data_array):
        cset = ax.contourf(
            xx, yy, data_array[:,frame_number].reshape(problem.n, problem.m).T,
            30, vmin = 0.0, vmax=data_max, cmap=reverse_purples
            )
        return cset

    ani = FuncAnimation(fig, update, all_data.shape[1], fargs=(all_data,))

    ani.save(target_dir + f"evol_scheme_{scheme}_dt_{dt}_N_{N}_M_{M}_psi0_{psi0}_psi1_{psi1}_ani.mp4")

if __name__ == "__main__":
    analyze_oversight()
    # trace()
    # plot_equilibrium()