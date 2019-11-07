#!/usr/bin/env python

from math import pi
from numpy import (
    finfo, asarray, zeros, linspace, set_printoptions, diag, ones,
    cos, sin
    )
from datetime import datetime

from initialize import problem_1D

from os import mkdir
from os.path import isdir

import fd_mod

set_printoptions(linewidth=500)

def get_params():

    # discretization parameters
    dt = 1e-2  # time discretization. Keep this number low
    N = 720 # inverse space discretization. Keep this number high!

    # model-specific parameters
    gamma = 1000.0  # drag
    beta = 1.0  # 1/kT
    m = 1.0  # mass

    E = 256.0 # energy scale of system

    psi1 = 0.0 # force on system by chemical bath B1
    psi2 = 0.0 # force on system by chemical bath B2

    n = 3.0 # number of minima in system potential

    mode = "spec_addif"

    return ( dt, N, gamma, beta, m, E, psi1, psi2, n, mode )

def save_data_reference(
    E, psi1, psi2, n, positions, p_ss, p_initial, p_equil,
    potential_at_pos, drift_at_pos, diffusion_at_pos, N
    ):

    target_dir = './master_output_dir/'
    data_filename = f'/reference_E_{E}_psi1_{psi1}_psi2_{psi2}_n_{n}_outfile.dat'
    data_total_path = target_dir + data_filename

    if not isdir(target_dir):
        print("Target directory doesn't exist. Making it now")
        mkdir(target_dir)

    with open(data_total_path, 'w') as dfile:
        for i in range(N):
            dfile.write(
                f"{positions[i]:.15e}"
                + "\t" + f"{p_ss[i]:.15e}"
                + '\t' + f"{p_initial[i]:.15e}"
                + '\t' + f"{p_equil[i]:.15e}"
                + '\t' + f"{potential_at_pos[i]:.15e}"
                + '\t' + f"{drift_at_pos[i]:.15e}"
                + '\t' + f"{diffusion_at_pos[i]:.15e}"
                + '\n'
            )

def main():

    # unload parameters
    [ dt, N, gamma, beta, m, E, psi1, psi2, n, mode ] = get_params()

    # calculate derived discretization parameters
    dx = (2*pi)/N  # space discretization: total distance / number of points

    # # provide CSL criteria to make sure simulation doesn't blow up
    # if E == 0.0:
    #     time_check = 100000000.0
    # else:
    #     time_check = dx*m*gamma / (3*E)

    # if dt > time_check:
    #     print("!!!TIME UNSTABLE!!! No use in going on. Aborting...\n")
    #     exit(1)

    # how many time update steps before checking for steady state convergence
    # enforce steady state convergence check every unit time
    check_step = int(1.0/dt)

    print(f"Number of times before check = {check_step}")

    # set initial distribution to be the uniform distribution
    p_initial = ones(N, order="F")/N
    # initialize array which holds the steady state distribution
    p_ss = zeros(N, order="F")

    problem_object = problem_1D(
        n=N, E=E, num_minima=n, D=1./(m*gamma), psi=psi1+psi2, mode=mode
        )

    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Launching reference simulation...")
    # fd_mod.fdiff.get_steady_ft(
    #     dt, problem_object.L, check_step, p_initial, p_ss, problem_object.n
    # )
    fd_mod.fdiff.get_steady_gl10(
        dt, problem_object.L, check_step, p_initial, p_ss, problem_object.n
    )
    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Reference simulation done!")

    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Processing data...")

    # checks to make sure nothing went weird
    # check non-negativity of distribution
    assert (p_ss >= 0.0).all(), \
        "ABORT: Probability density has negative values!"
    # check normalization of distribution
    assert (abs(p_ss.sum() - 1.0) <= finfo('float32').eps), \
        "ABORT: Probability density is not normalized!"

    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Processing finished!")

    # write to file
    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving data...")
    save_data_reference(
        E, psi1, psi2, n, problem_object.theta, p_ss, p_initial, problem_object.p_equil,
        problem_object.Epot, problem_object.mu, problem_object.D, N
        )
    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving completed!")

    print("Exiting...")

if __name__ == "__main__":
    main()