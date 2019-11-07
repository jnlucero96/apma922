#!/usr/bin/env python3

from math import pi
from numpy import array, linspace, zeros, ones, diag, sin, cos, exp

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

        self.mu = self.drift_1d() # drift vector
        self.Epot = self.potential() # potential landscape

        self.p_equil = exp(-self.Epot)/exp(-self.Epot).sum()

        # select method of solution
        if (self.mode.lower() == "cs_addif"):
            self.L = self.cs_addif_1D_init()

    def potential(self):
        return 0.5*self.Edagger*(1.0-cos(self.num_minima*self.theta))

    # define drift vector
    def drift_1d(self):
        return -(
            0.5*self.Edagger*self.num_minima*sin(self.num_minima*self.theta)
            -self.psi
            )

    # ====================== ADVECTION OPERATORS =============================

    # # Central-Space Advection
    # def cs_advective_1D_init():
    #     return L

    # # Central-Space Diffusion
    # def cs_diffusion_1D_init():
    #     return L

    # ====================== DIFFUSION OPERATORS =============================

    # # Spectral-Space Diffusion
    # def spec_diffusion_1D_init():
    #     return L

    # # Spectral-Space Diffusion
    # def spec_diffusion_1D_init():
    #     return L

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

    # # Spectral-Space Advection-Diffusion
    # def spec_addif_1d_init():
    #     return L

# =================== TWO DIMENSIONAL PROBLEMS ==============================

# # define drift vector
# def drift_2D():
#     return 0

# # define the diffusion coefficient
# def diffusion_2D():
#     return 0

# # Central-Space Advection
# def cs_advective_2D_init():
#     return 0

# # Central-Space Diffusion
# def cs_diffusion_2D_init():
#     return 0

# # Spectral-Space Diffusion
# def spec_diffusion_2D_init():
#     return 0

# # Spectral-Space Diffusion
# def spec_diffusion_2D_init():
#     return 0

# # Central-Space Advection-Diffusion
# def cs_addif_2D_init():
#     return 0

# # Spectral-Space Advection-Diffusion
# def spec_addif_2d_init():
#     return 0