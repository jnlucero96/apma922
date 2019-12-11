#!/usr/bin/env python3

from math import pi
from numpy import (
    array, linspace, zeros, ones, diag, sin, cos, exp, arange, tan, kron,
    eye, union1d, where, meshgrid
    )
from scipy.linalg import toeplitz

# =================== ONE DIMENSIONAL PROBLEMS ==============================

class problem_1D(object):

    def __init__(
        self, n=360, E=2.0, num_minima=3.0, D=0.001, psi=0.0, mode="cs_addif"
        ):

        # unload the variables
        self.n = n # number of points
        self.Edagger = E # barrier heights
        self.num_minima = num_minima # periodicity of potential
        self.D = D*ones(self.n) # diffusion coefficient
        self.psi = psi

        self.mode = mode # method of solution

        # compute the derived variables
        # discretization
        self.dx = (2*pi)/self.n
        # grid
        self.theta = linspace(0.0, 2.0*pi-self.dx, self.n)

        self.mu = self.drift() # drift vector
        self.Epot = self.potential() # potential landscape

        self.p_equil = exp(-self.Epot)/exp(-self.Epot).sum()

        # select method of solution
        if (self.mode.lower() == "cs_advective"):
            self.L = self.cs_advective_1D_init()
        elif (self.mode.lower() == "spec_advective"):
            self.L = self.spec_advective_1D_init()
        elif (self.mode.lower() == "cs_diffusion"):
            self.L = self.cs_diffusion_1D_init()
        elif (self.mode.lower() == "spec_diffusion"):
            self.L = self.spec_diffusion_1D_init()
        elif (self.mode.lower() == "cs_addif"):
            self.L = self.cs_addif_1D_init()
        elif (self.mode.lower() == "spec_addif"):
            self.L = self.spec_addif_1D_init()
        else:
            print("Mode input not understood! Exiting now")
            exit(1)

    def potential(self):
        return 0.5*self.Edagger*(1.0-cos(self.num_minima*self.theta))

    # define drift vector
    def drift(self):
        return -(
            0.5*self.Edagger*self.num_minima*sin(self.num_minima*self.theta)
            -self.psi
            )

    # ====================== ADVECTION OPERATORS =============================

    # Central-Space Advection
    def cs_advective_1D_init(self):

        D1 = zeros((self.n,self.n), order="F")

        # define first-derivative matrix based on centered difference
        D1[...] = (
            -diag(self.mu[:-1],-1)
            + diag(zeros(self.n))
            + diag(self.mu[1:],1)
            )
        D1[0,-1] = -self.mu[-1]; D1[-1,0] = self.mu[0] #enforce periodicity

        return D1

    # Central-Space Advection
    def spectral_diffusion_1D_init(self):

        D1 = zeros((self.n,self.n),order="F")

        jj = arange(1,self.n)

        col = zeros(self.n)
        col[1:] = (0.5*(-1.0)**jj)/tan(0.5*jj*self.dx)
        row = zeros(self.n)
        row[0] = col[0]
        row[1:] = col[self.n-1:0:-1]
        D1[...] = toeplitz(col, row)*self.mu

        return D1

    # ====================== DIFFUSION OPERATORS =============================

    # Central-Space Diffusion
    def cs_diffusion_1D_init(self):
        D2 = zeros((self.n,self.n), order="F")

        # define central-space differentiation matrix
        D2[...] = (
            diag(ones(self.n-1),-1)
            + diag(-2.0*ones(self.n))
            + diag(ones(self.n-1),1)
            )
        D2[0,-1] = 1.0; D2[-1,0] = 1.0 # enforce periodicity

        # scale the matrices appropriately
        D2 /= self.dx**2

        return D2

    # Spectral-Space Diffusion
    def spec_diffusion_1D_init(self):
        jj = arange(1,self.n)

        D2 = zeros((self.n,self.n),order="F")

        column = zeros(self.n)
        column[0] = -((pi**2)/(3*(self.dx**2))+(1./6))
        column[1:] = -0.5*((-1)**jj)/(sin(0.5*jj*self.dx)**2)
        D2[...] = toeplitz(column)

        return D2

    # ================= ADVECTION-DIFFUSION OPERATORS ======================

    # Central-Space Advection-Diffusion
    def cs_addif_1D_init(self):

        # initialize the differentiation matrices
        D1 = zeros((self.n,self.n), order="F")
        D2 = zeros((self.n,self.n), order="F")

        # define first-derivative matrix based on centered difference
        D1[...] = -diag(self.mu[:-1],-1) + diag(zeros(self.n)) + diag(self.mu[1:],1)
        D1[0,-1] = -self.mu[-1]; D1[-1,0] = self.mu[0] #enforce periodicity

        # define central-space differentiation matrix
        D2[...] = diag(ones(self.n-1),-1) + diag(-2.0*ones(self.n)) + diag(ones(self.n-1),1)
        D2[0,-1] = 1.0; D2[-1,0] = 1.0 # enforce periodicity

        # scale the matrices appropriately
        D1 /= 2.0*self.dx
        D2 /= self.dx**2

        return self.D*(-D1+D2)

    # Spectral-Space Advection-Diffusion
    def spec_addif_1D_init(self):

        D1 = zeros((self.n,self.n),order="F")
        D2 = zeros((self.n,self.n),order="F")

        jj = arange(1,self.n)

        col = zeros(self.n)
        col[1:] = (0.5*(-1.0)**jj)/tan(0.5*jj*self.dx)
        row = zeros(self.n)
        row[0] = col[0]
        row[1:] = col[self.n-1:0:-1]
        D1[...] = toeplitz(col, row)*self.mu

        column = zeros(self.n)
        column[0] = -((pi**2)/(3*(self.dx**2))+(1./6))
        column[1:] = -0.5*((-1)**jj)/(sin(0.5*jj*self.dx)**2)
        D2[...] = toeplitz(column)

        return self.D*(-D1+D2)

# =================== TWO DIMENSIONAL PROBLEMS ==============================

class problem_2D(object):

    def __init__(
        self, x0=0.0, xn=2*pi,
        y0=0.0, ym=2*pi,
        n=360, m=360,
        E0=2.0, Ec=8.0, E1=2.0,
        num_minima0=3.0, num_minima1=3.0,
        D=0.001, psi0=0.0, psi1=0.0
        ):

        # unload the variables
        # define the end points of the intervals
        self.x0 = x0
        self.xn = xn
        self.y0 = y0
        self.ym = ym
        self.n = n # number of points in x
        self.m = m # number of points in y
        # barrier heights
        self.E0 = E0
        self.Ec = Ec
        self.E1 = E1
        # periodicity of potential
        self.num_minima0 = num_minima0
        self.num_minima1 = num_minima1

        # diffusivity
        self.D = D

        # nonequilibrium forces
        self.psi0 = psi0
        self.psi1 = psi1

        # compute the derived variables

        # discretization
        self.L0 = xn-x0
        self.L1 = ym-y0
        self.dx = self.L0/self.n
        self.dy = self.L1/self.m

        # grid
        self.theta0 = linspace(x0, xn-self.dx, self.n)
        self.theta1 = linspace(y0, ym-self.dy, self.m)

        # drift matrices
        self.mu1 = self.drift1()
        self.mu2 = self.drift2()
        self.dmu1 = self.ddrift1()
        self.dmu2 = self.ddrift2()

        # potential landscape
        self.Epot = self.potential()

        # define equilibrium distribution
        self.p_equil = exp(-self.Epot)/exp(-self.Epot).sum()

    # define the potential V
    def potential(self):
        return 0.5*(
            self.E0*(1.0-cos(self.num_minima0*self.theta0[:,None]))
            + self.Ec*(1.0-cos(self.theta0[:,None]-self.theta1[None,:]))
            + self.E1*(1.0-cos(self.num_minima1*self.theta1[None,:]))
            )

    # define drift vector mu_{1}
    def drift1(self):
        return -(
            0.5*(self.Ec*sin(self.theta0[:,None]-self.theta1[None,:])
            + self.E0*self.num_minima0*sin(self.num_minima0*self.theta0[:,None])
            ) - self.psi0)

    # define drift vector mu_{2}
    def drift2(self):
        return -(
            0.5*(-self.Ec*sin(self.theta0[:,None]-self.theta1[None,:])
            + self.E1*self.num_minima1*sin(self.num_minima1*self.theta1[None,:])
            ) - self.psi1)

    # additional derivatives of the drift vectors above
    def ddrift1(self):
        return -(
            0.5*(self.Ec*cos(self.theta0[:,None]-self.theta1[None,:])
            + self.E0*(self.num_minima0**2)*cos(self.num_minima0*self.theta0[:,None])
            ))
    def ddrift2(self):
        return -(
            0.5*(self.Ec*cos(self.theta0[:,None]-self.theta1[None,:])
            + self.E1*(self.num_minima1**2)*cos(self.num_minima1*self.theta1[None,:])
            ))