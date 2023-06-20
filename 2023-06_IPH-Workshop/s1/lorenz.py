#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:34:42 2018

Module containing functions to deal with the Lorenz system
"""
import numpy as np
import matplotlib.pyplot as plt


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


def plot2DCurve(x, y, figName, style='-', label=None,
                logX=False, logY=False, xLabel='X', yLabel='', showFig=False):
    """
    Plot a 2D curve in a given figure

    Parameters
    ----------
    x : vector
        The x-data to plot
    y : vector
        The y-data to plot
    figName : str
        The name of the figure
    style : str
        The style of the curve (can be '-' for line, '--' for dashes, ':' for
        points, ...)
    label : str
        The label of the curve (default is None, do not register any label)
    logX : boolean
        Use logarithmic scale for x axis (default=False)
    logY : boolean
        Use logarithmic scale for y axis (default=False)
    xLabel : str
        Label for the x axis (default='X')
    yLabel : str
        Label for the y axis (default='')
    showAnim : bool
        If True, show the plot at the end of the function call
        (with a call to plt.show()).

    """
    plt.figure(figName)
    plt.plot(x, y, style, label=label)
    if logX:
        plt.xscale('log')
    if logY:
        plt.yscale('log')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.grid(True)
    if showFig:
        plt.show()