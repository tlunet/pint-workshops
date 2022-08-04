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


def nievergelt(u0, f, g, tBeg, tEnd, N):

    times = np.linspace(tBeg, tEnd, num=N+1)

    # Coarse propagation
    coarse = [u0]*(N+1)
    for i in range(N):
        coarse[i+1] = g(coarse[i])

    # Shooting solutions
    shoot = [[] for _ in range(N)]
    # -- first shoot is fine solve
    shoot[0].append(f(u0))
    offset = 0.5
    for i in range(N-1):
        shoot[i+1].append(f((1+offset)*u0))
        shoot[i+1].append(f((1-offset)*u0))
