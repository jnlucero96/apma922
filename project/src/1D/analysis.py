#!/usr/bin/env python3

from numpy import loadtxt

from matplotlib.style import use

use("dark_background")

from pltconfig import *

def get_params():

    # discretization parameters
    dt = 5e-1  # time discretization. Keep this number low
    N = 360 # inverse space discretization. Keep this number high!

    # model-specific parameters
    gamma = 1000.0  # drag
    beta = 1.0  # 1/kT
    m = 1.0  # mass

    E = 2.0 # energy scale of system

    psi1 = 0.0 # force on system by chemical bath B1
    psi2 = 0.0 # force on system by chemical bath B2

    n = 3.0 # number of minima in system potential

    mode = "spec_addif"

    dim = 1

    return ( dt, N, gamma, beta, m, E, psi1, psi2, n, mode, dim )

def analyze():

    [ dt, N, gamma, beta, m, E, psi1, psi2, n, mode, dim ] = get_params()

    target_dir = './master_output_dir/'
    data_filename = f'/ref_{dim}D_E_{E}_psi1_{psi1}_psi2_{psi2}_n_{n}_outfile.dat'
    data_total_path = target_dir + data_filename

    positions, p_ss, p_equil = loadtxt(
        data_total_path, usecols=(0,1,3), unpack=True
        )

    fig, ax = subplots(1,1)
    ax.set_ylabel(r"$P(\theta)$", fontsize=28)
    ax.set_xlabel(r"$\theta$", fontsize=28)

    ax.plot(positions, p_equil, "r--", label=r"$\pi^{\mathrm{eq}}$")
    ax.plot(positions, p_ss, "o", ms=7.0, label=r"$p^{\mathrm{ss}}$")

    ax.legend(loc=0, prop={"size": 18})
    ax.ticklabel_format(style="sci",scilimits=(0,0))

    fig.tight_layout()
    fig.savefig("figure.pdf")

if __name__ == "__main__":
    analyze()



