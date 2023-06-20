#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:16:35 2022

@author: telu
"""
import numpy as np


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


def findClosestIndex(value, lValues):
    """Find the indexes of the element in a list closest to a given value

    Parameters
    ----------
    value : scalar
        The targeted value.
    lValues : TYPE
        The list of values from which the closest values have to be extracted.

    Returns
    -------
    i1, i2
        A list containing the two indexes of the closest values,
        sorted in ascending value.
    """
    lValues = np.asarray(lValues)
    diff = np.abs(lValues-value)
    i1 = np.argmin(diff)
    diff[i1] = np.inf
    i2 = np.argmin(diff)
    return sorted([i1, i2])


def nievergelt(u0, fineSolver, coarseSolver, tBeg, tEnd, N,
               Mn, delta):
    """
    Run the Nievergelt algorithm

    Parameters
    ----------
    u0 : TYPE
        DESCRIPTION.
    fineSolver : TYPE
        DESCRIPTION.
    coarseSolver : TYPE
        DESCRIPTION.
    tBeg : TYPE
        DESCRIPTION.
    tEnd : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    Mn : TYPE
        DESCRIPTION.
    delta : TYPE
        DESCRIPTION.

    Returns
    -------
    uNiev : TYPE
        DESCRIPTION.
    maxErr : TYPE
        DESCRIPTION.
    uCoarse : TYPE
        DESCRIPTION.
    uFine : TYPE
        DESCRIPTION.
    shoot : TYPE
        DESCRIPTION.
    listP : TYPE
        DESCRIPTION.
    times : TYPE
        DESCRIPTION.

    """

    # Define the time decomposition
    times = np.linspace(tBeg, tEnd, num=N+1)

    # Compute fine solution on each points (for comparison)
    uFine = [u0 for _ in range(N+1)]
    for n in range(N):
        uFine[n+1] = fineSolver(uFine[n], times[n], times[n+1])[-1]

    # Coarse propagation
    uCoarse = [u0 for _ in range(N+1)]
    for n in range(N):
        uCoarse[n+1] = coarseSolver(uCoarse[n], times[n], times[n+1])[-1]

    # Shooting solutions
    shoot = [[] for _ in range(N)]
    # -- first shoot is fine solve
    uLeft = uCoarse[0]
    uInner = fineSolver(u0, times[0], times[1])
    uRight = uInner[-1]
    shoot[0].append({"left": uLeft, "right": uRight, "inner": uInner})
    # -- multiple shooting (n=2)
    shootGrid = np.linspace(-delta/2, delta/2, num=Mn)
    for n in range(1, N):
        for offset in shootGrid:
            uLeft = uCoarse[n] + offset
            uInner = fineSolver(uLeft, times[n], times[n+1])
            uRight = uInner[-1]
            shoot[n].append({"left": uLeft, "right": uRight, "inner": uInner})

    # Corrected solutions
    uNiev = [u0 for _ in range(N+1)]
    # -- take the fine solution for the first time sub-interval
    uNiev[1] = shoot[0][0]["right"]
    # -- linear interpolation for the other time sub-intervals
    listP = []
    for n in range(1, N):
        # -- corrected solution from previous sub-interval
        u1 = uNiev[n]
        # -- list of initial shooting values for current interval
        listInit = [sol["left"] for sol in shoot[n]]
        # -- find closest value indices
        m1, m2 = findClosestIndex(u1, listInit)
        # -- compute p for linear interpolation
        uS1 = shoot[n][m1]["left"]
        uS2 = shoot[n][m2]["left"]
        p = (u1 - uS2)/(uS1 - uS2)
        listP.append(p)
        # -- compute corrected solution
        uNiev[n+1] = p*shoot[n][m1]["right"] + (1-p)*shoot[n][m2]["right"]

    # Transform into Numpy arrays and compute max error with fine solution
    uFine = np.array(uFine)
    uNiev = np.array(uNiev)
    maxErr = np.linalg.norm(uNiev-uFine, ord=np.inf)

    return uNiev, maxErr, uCoarse, uFine, shoot, listP, times
