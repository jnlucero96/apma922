#!/usr/bin/env python

from math import pi
from numpy import finfo, zeros, set_printoptions, ones
from datetime import datetime

from initialize import problem_1D, problem_2D

from os import mkdir
from os.path import isdir

import spectral_mod

from sys import stderr

set_printoptions(linewidth=500)

scheme_to_name = {
    1: "Forward Euler",
    2: "IMEX",
    3: "RK2",
    4: "RK4",
    5: "IFRK4",
    6: "ETDRK4",
    7: "GL04",
    8: "GL06",
    9: "GL08",
    10: "GL10"
}

target_dir = './master_output_dir/'

def get_params():

    # discretization parameters
    dt = 5e-1  # time discretization. Keep this number low
    N = 60  # inverse space discretization. Keep this number high!
    M = 60  # inverse space discretization. Keep this number high!

    psi1 = 0.0  # force on system by chemical bath B1
    psi2 = 0.0  # force on system by chemical bath B2

    steady = True # evolve to steady-state distribution

    # solver schemes:
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

    scheme = 5

    return ( dt, N, M, psi1, psi2, steady, scheme )

def save_data_reference(
    scheme, dt, psi0, psi1, p_ss, p_equil, N, M
    ):

    data_filename = f'/ref_scheme_{scheme}_dt_{dt}_N_{N}_M_{M}_psi0_{psi0}_psi1_{psi1}_outfile.dat'
    data_total_path = target_dir + data_filename

    if not isdir(target_dir):
        print("Target directory doesn't exist. Making it now")
        mkdir(target_dir)

    with open(data_total_path, 'w') as dfile:

        for i in range(N):
            for j in range(M):
                dfile.write(
                    f"{p_ss[i,j]:.15e}"
                    + '\t' + f"{p_equil[i,j]:.15e}"
                    + '\n'
                )

def save_data_evolution(
    scheme, dt, psi0, psi1, p_trace, nchecks, N, M
    ):

    data_filename = f'/evol_scheme_{scheme}_dt_{dt}_N_{N}_M_{M}_psi0_{psi0}_psi1_{psi1}_outfile.dat'
    data_total_path = target_dir + data_filename

    if not isdir(target_dir):
        print("Target directory doesn't exist. Making it now")
        mkdir(target_dir)

    with open(data_total_path, 'w') as dfile:

        for i in range(N):
            for j in range(M):
                for step in range(nchecks):
                    dfile.write(f"{p_trace[i,j,step]:.15e}\t")
                dfile.write("\n")

def main():

    # unload parameters
    [ dt, N, M, psi1, psi2, steady, scheme ] = get_params()

    # enforce steady state convergence check every unit time
    check_step = int(1.0/dt)

    print(f"Using Spectral Space {scheme_to_name[scheme]} method")
    print(f"Number of times before check = {check_step}")

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Prepping reference simulation..."
        )

    # define the problem
    problem = problem_2D(
        x0=0.0, xn=2.0*pi, y0=0.0, ym=2.0*pi, n=N, m=M,
        E0=2.0, Ec=8.0, E1=2.0, num_minima0=3.0, num_minima1=3.0,
        D=0.001, psi0=psi1, psi1=psi2
    )

    # set initial distribution to be the uniform distribution
    p_initial = ones((problem.n, problem.m), order="F")/(problem.n*problem.m)
    # initialize array which holds the steady state distribution
    p_ss = zeros((problem.n, problem.m), order="F")

    drift1 = zeros((problem.n, problem.m), order="F")
    ddrift1 = zeros((problem.n, problem.m), order="F")
    drift2 = zeros((problem.n, problem.m), order="F")
    ddrift2 = zeros((problem.n, problem.m), order="F")
    drift1[...] = problem.mu1
    drift2[...] = problem.mu2
    ddrift1[...] = problem.dmu1
    ddrift2[...] = problem.dmu2

    if not steady:
        T = 7_000.0 # time to evolve for
        nsteps = int(T/dt)

        nchecks = int(T/50.0)
        p_trace = zeros((problem.n, problem.m, nchecks), order="F")
        p_trace[:,:,0] = p_initial

        check_step = int(50.0/dt)
        refarray = None
    else:
        T = None; nsteps=None; nchecks=None; p_trace=None

        refarray = zeros(1, order="F")

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Starting reference simulation..."
        )

    start_time = datetime.now() # record starting time
    if not steady:
        spectral_mod.fft_solve.get_spectral_fwd(
            nsteps, dt, scheme, check_step, problem.D, problem.dx, problem.dy,
            drift1, ddrift1, drift2, ddrift2,
            p_initial, p_trace, nchecks, problem.n, problem.m
            )
    else:
        spectral_mod.fft_solve.get_spectral_steady(
            dt, scheme, check_step, problem.D, problem.dx, problem.dy,
            drift1, ddrift1, drift2, ddrift2,
            p_initial, p_ss, refarray, problem.n, problem.m
            )
    end_time = datetime.now() # record ending time

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Reference simulation done!"
        )

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Processing data...")

    if not steady:
        # set all small enough numbers to zero
        p_trace[p_trace.__abs__() <= finfo("float64").eps] = 0.0

        if not ((p_trace >= 0.0).all()):
            print(
                "Probability density has non-negligible negative values!",
                file=stderr
                )
        if not (((p_trace.sum(axis=(0,1)) - 1.0).__abs__() <= finfo('float32').eps).all()):
            print("Probability density is not normalized!", file=stderr)
    else:
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

    if not steady:
        save_data_evolution(
            scheme, dt, problem.psi0, problem.psi1, p_trace, nchecks, problem.n, problem.m
        )
    else:
        save_data_reference(
            scheme, dt, problem.psi0, problem.psi1, p_ss, problem.p_equil,
            problem.n, problem.m
        )
    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving completed!")

    if steady:
        print(
            f"Total time simulation time elapsed (minutes): {(end_time-start_time).total_seconds()/60.0}\n"
            + f"Number of checks: {refarray[0]}\n"
            + f"Max inf-norm error of solution: {(p_ss-problem.p_equil).__abs__().max()}"
            )
        with open(target_dir + f"/sp_oversight_scheme_{scheme}_dt_{dt}_file.dat", "a") as datfile:
            datfile.write(
                f"{problem.n}\t"
                + f"{problem.m}\t"
                + f"{refarray[0]}\t"
                + f"{(end_time-start_time).total_seconds()/60.0}\t"
                + f"{(p_ss-problem.p_equil).__abs__().max()}\n"
                )
    else:
        print(
            f"Total time simulation time elapsed (minutes): {(end_time-start_time).total_seconds()/60.0}"
            )

    print("Exiting...")


if __name__ == "__main__":
    main()
