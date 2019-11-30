#!/usr/bin/env python

from math import pi
from numpy import (
    finfo, asarray, zeros, linspace, set_printoptions, diag, ones,
    cos, sin, kron
)
from datetime import datetime

from initialize import problem_1D, problem_2D

from os import mkdir
from os.path import isdir

import spectral_mod

from sys import stderr

set_printoptions(linewidth=500)

def get_params():

    # discretization parameters
    dt = 1e-2  # time discretization. Keep this number low
    N = 80  # inverse space discretization. Keep this number high!
    M = 80  # inverse space discretization. Keep this number high!

    psi1 = 0.0  # force on system by chemical bath B1
    psi2 = 0.0  # force on system by chemical bath B2

    return ( dt, N, M, psi1, psi2 )

def save_data_reference(
    dt, psi0, psi1, p_ss, p_initial, p_equil,
    potential_at_pos, mu1, mu2, N, M
    ):

    target_dir = './master_output_dir/'
    data_filename = f'/ref_dt_{dt}_N_{N}_M_{M}_psi0_{psi0}_psi1_{psi1}_outfile.dat'
    data_total_path = target_dir + data_filename

    if not isdir(target_dir):
        print("Target directory doesn't exist. Making it now")
        mkdir(target_dir)

    with open(data_total_path, 'w') as dfile:

        for i in range(N):
            for j in range(M):
                dfile.write(
                    f"{p_ss[i,j]:.15e}"
                    + '\t' + f"{p_initial[i,j]:.15e}"
                    + '\t' + f"{p_equil[i,j]:.15e}"
                    + '\t' + f"{potential_at_pos[i,j]:.15e}"
                    + '\t' + f"{mu1[i,j]:.15e}"
                    + '\t' + f"{mu2[i,j]:.15e}"
                    + '\n'
                )

def main():

    # unload parameters
    [ dt, N, M, psi1, psi2 ] = get_params()

    # enforce steady state convergence check every unit time
    check_step = int(1.0/dt)

    print(f"Number of times before check = {check_step}")

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Prepping reference simulation..."
        )

    # set initial distribution to be the uniform distribution
    p_initial = ones((N, M), order="F")/(N*M)
    # initialize array which holds the steady state distribution
    p_ss = zeros((N, M), order="F")

    problem = problem_2D(
        x0=0.0, xn=2.0*pi, y0=0.0, ym=2.0*pi, n=N, m=M, 
        E0=2.0, Ec=8.0, E1=2.0, num_minima0=3.0, num_minima1=3.0,
        D=0.001, psi0=psi1, psi1=psi2
    )

    drift1 = zeros((N, M), order="F")
    ddrift1 = zeros((N, M), order="F")
    drift2 = zeros((N, M), order="F")
    ddrift2 = zeros((N, M), order="F")
    drift1[...] = problem.mu1
    drift2[...] = problem.mu2
    ddrift1[...] = problem.dmu1
    ddrift2[...] = problem.dmu2

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Starting reference simulation..."
        )

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

    start_time = datetime.now() # record starting time
    spectral_mod.fft_solve.get_spectral_steady(
        dt, 6, check_step, problem.D, problem.dx, problem.dy,
        drift1, ddrift1, drift2, ddrift2,
        p_initial, p_ss, problem.n, problem.m
        )
    end_time = datetime.now() # record ending time

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Reference simulation done!"
        )

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Processing data...")

    # set all small enough numbers to zero
    p_ss[p_ss.__abs__() <= finfo("float64").eps] = 0.0

    if not ((p_ss >= 0.0).all()):
        print(
            "Probability density has non-negligible negative values!",
            file=stderr
            )
    if not ((abs(p_ss.sum() - 1.0) <= finfo('float32').eps)):
        print("Probability density is not normalized!", file=stderr)

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Processing finished!"
        )

    # write to file
    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving data...")
    save_data_reference(
        dt, problem.psi0, problem.psi1, p_ss, p_initial, problem.p_equil,
        problem.Epot, problem.mu1, problem.mu2, problem.n, problem.m
    )
    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving completed!")


    print(
        f"Total time simulation time elapsed (minutes): {(end_time-start_time).total_seconds()/60.0}"
        )
    print(
        f"Max inf-norm error of solution: {(p_ss-problem.p_equil).__abs__().max()}"
        )
    print("Exiting...")


if __name__ == "__main__":
    main()
