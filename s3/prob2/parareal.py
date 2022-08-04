#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:35:35 2022

@author: telu
"""
import numpy as np
import scipy.linalg as spl


def lorenzOperator(x, y, z, sigma=10, rho=28, beta=8/3):
    """
    Evaluate the Lorenz operator on a given position vector
    :math:`(x,y,z)` as follow:

    .. math::
        f(x,y,z) = \\begin{pmatrix} \\sigma(y-x) \\\\  x(\\rho-z)-y \\\\
        xy-\\beta z \\end{pmatrix}

    Parameters
    ----------
    x : float
        The :math:`x` position
    y : float
        The :math:`y` position
    z : float
        The :math:`z` position
    sigma : float
        The :math:`\\sigma` parameter (default=10)
    rho : float
        The :math:`\\rho` parameter (default=28)
    beta : float
        The :math:`\\beta` parameter (default=8/3)

    Returns
    -------
    fxyz : list of 3 elements
        The components of the evaluated operator
    """
    return sigma*(y-x), x*(rho-z)-y, x*y-beta*z


def forwardEuler(f, t0, tEnd, u0, nStep):
    """
    Uses the Forward Euler method to solve the system of first order
    ordinary differential equations

    .. math::
        \\frac{du}{dt} = f(t,u)

    with :math:`u` a vector of size :math:`N_{dof}`.
    The final solution is computed using :math:`N_{step}` time steps.
    At each time steps, the updated solution is computed using the formula :

    .. math::
        u(t+dt) = u(t) + dt \\times f(t,u(t)),

    where :math:`dt=\\frac{t_{end}-t_0}{N_{step}}`.

    Parameters
    ----------
    f : function
        The :math:`f` operator, as a function that take a scalar **t** and
        a numpy vector **u** as argument,
        and returns a vector of the same size as **u**.
    t0 : float
        The time of the initial solution
    tEnd : float
        The time of the final solution
    u0 : numpy vector or list
        The initial solution
    nStep : int
        The number of time step to be performed

    Returns
    -------
    t : numpy vector of size :math:`N_{step}+1`
        The discrete times from **t0** to **tEnd** when the solution was
        computed (including the initial time).
    u : numpy matrix of size :math:`(N_{dof} \\times N_{step}+1)`
        The solution of the ODE at each time steps, including the initial time.

    """
    # Create output variables
    nDOF = np.asarray(u0).size
    u = np.zeros((nDOF, nStep+1))
    t = np.linspace(t0, tEnd, nStep+1)

    # Store initial solution and time
    u[:, 0] = u0

    # Compute time-step
    dt = (tEnd-t0)/nStep

    # Loop on every time-step
    for i in range(nStep):
        # Evaluation of the operator
        u[:, i+1] = f(t[i], u[:, i])
        # Multiplication by time-step
        u[:, i+1] *= dt
        # Addition of previous step solution
        u[:, i+1] += u[:, i]

    return t, u


def finDiffMatrixD2C2(J, L):
    """Compute the finite-difference matrix for the second derivative
    in space (D2) using centered finite differences of order 2 (C2).

    Parameters
    ----------
    J : int
        Number of mesh point in space (excluding boundary conditions)
    L : float
        Length of the domain (including boundary conditions)
    """
    h = L/(J+1)
    A = spl.toeplitz([-2., 1.]+(J-2)*[0.])
    A /= h**2
    return A


def finDiffMatrixD1U1(J, L):
    """Compute the finite-difference matrix for the first derivative
    in space (D1) using first order Upwind finite differences (U1),
    and periodic boundary conditions.

    Parameters
    ----------
    J : int
        Number of mesh point in space (excluding right point)
    L : float
        Length of the domain (including right point)
    """
    h = L/J
    A = spl.circulant([1, -1.]+(J-2)*[0.])
    A /= h
    return A


def backwardEulerLin(A, u0, b, tBeg, tEnd, N):
    """
    Solve a linear system of ODE using Backward Euler.
    Considering the problem

    .. math::
        \\frac{dU}{dt} = AU + b(t),

    computes the numerical solution with Backward Euler, between **tBeg**
    and **tEnd**, with **u0** the initial solution, using **N** steps.

    Parameters
    ----------
    A : matrix of size JxJ
        The matrix of the linear system
    u0 : vector of size J
        The initial solution
    b : function
        The source term
    tBeg : float
        The initial time of the solution
    tEnd : float
        The final time of the solution
    N : int
        The number of numerical time steps

    Returns
    -------
    u : matrix of size JxN
        The solution at each time steps (including initial solution)
    times : vector of size N
        The times of the solutions
    """
    dt = (tEnd-tBeg)/N
    times = np.linspace(tBeg, tEnd, N+1)
    u0 = np.asarray(u0)
    J = u0.size
    u = np.zeros((J, N+1), dtype=u0.dtype, order='F')
    u[:, 0] = u0
    R = np.copy(A)
    R *= -dt
    R += np.eye(J)
    for i, t in enumerate(times[1:]):
        u[:, i+1] = np.linalg.solve(R, u[:, i] + dt*b(t))
    return times, u


def parareal(u0, tBeg, tEnd, N,
             F, G, K):
    """
    Run the Parareal algorithm

    Parameters
    ----------
    u0 : vector of size nDOF
        The initial solution
    tBeg : float
        Initial time of simulation.
    tEnd : TYPE
        End time of simulation.
    N : int
        Number of time subintervals
    F : function
        The fine propagator on one sub-interval, which signature has to be
        u = F(u0, tBeg, tEnd), with u the solution at the end of the time
        sub-interval
    G : function
        The coarse propagator on one sub-interval, which signature has to be
        u = F(u0, tBeg, tEnd), , with u the solution at the end of the time
        sub-interval
    K : int
        Number of Parareal iteration (K=0 => only initialization)

    Returns
    -------
    Uk : list of (K+1) matrices of size nDOF x (N+1)
        The Parareal solution at each time subinterval interfaces
    uF : matrix of size nDOF x (N+1)
        The fine solution at each time subinterval interfaces
    err : vector of size (k+1)
        The error (:math:`L_\infty` in time, :math:`L_2` in space) of
        the Parareal solution, comparing to the fine solution.
    """
    print("Running Parareal")
    # Define the time decomposition
    times = np.linspace(tBeg, tEnd, num=N+1)

    print(" -- computing reference fine solution")
    # Compute fine solution on each points (for comparison)
    uFine = [u0 for _ in range(N+1)]
    for n in range(N):
        uFine[n+1] = F(uFine[n], times[n], times[n+1])
    uFine = np.array(uFine)

    # Initial vector and construction of list of Parareal solutions
    nDOF = np.size(u0)
    Uk = [np.zeros((N+1, nDOF)) for k in range(K+1)]

    # Values for U^k_0
    for k in range(K+1):
        Uk[k][0] = u0

    print(" -- initialization")
    # Initialization of Parareal
    for n in range(N):
        Uk[0][n+1] = G(Uk[0][n], times[n], times[n+1])

    # Loop on each Parareal iterations
    for k in range(K):
        print(' -- iteration {}/{}'.format(k+1, K))
        # Loop on each time subintervals
        for n in range(N):
            print(' -- sub-interval {}/{}'.format(n+1, N))
            Fk0 = F(Uk[k][n], times[n], times[n+1])
            Gk1 = G(Uk[k+1][n], times[n], times[n+1])
            Gk0 = G(Uk[k][n], times[n], times[n+1])
            Uk[k+1][n+1] = Fk0 + Gk1 - Gk0

    # Error computation
    err = [np.max(np.linalg.norm(Uk[k] - uFine, axis=0)) for k in range(K+1)]
    err = np.array(err)

    return Uk, uFine, err, times
