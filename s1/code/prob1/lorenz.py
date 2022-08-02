#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:34:42 2018

Module containing functions to deal with the Lorenz system
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


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


def plot2DCurve(x, y, figName, style='-', label=None,
                logX=False, logY=False, xLabel='X', yLabel='', showFig=True):
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


def plotAnimated3DCurve(lData, lTimes, figName,
                        wholeTraj=True, deltaData=1, lFixedPoints=None,
                        deltaFrame=20, showAnim=True):
    """
    Display an animation plot of a 3D trajectory on a given figure of one of
    more 3D vectors depending on time.
    The trajectory can be plotted in two way:

    - a line displaying all the trajectory in time, implementing it at each
      frame (when the **wholeTraj** parameter is set to **True**)
    - one free point wandering allong the trajectory, which position is
      updated at each frame (when the **wholeTraj** parameter is set to
      **False**)

    Parameters
    ----------
    lData : list of 3D vector
        The list containing all the trajectory vector of size
        :math:`(3 \\times N_{step}+1)` where :math:`N_{step}` is the number
        of time-step of the trajectory.
    lTimes : list or numpy vector
        The times of each trajectory position, of size :math:`N_{step}+1`.
    figName : str
        The name of the figure for the animation.
    wholeTraj : bool
        Display the trajectory (if **True**) or only a point with the current
        position (if **False**).
    deltaData : int
        The step for the trajectory position to be displayed.
    lFixedPoints : list of list(s) containing 3 float
        The position of fixed point(s) to plot on the animation.
    deltaFrame : int
        The number of millisecond before refreshing each frame.
    showAnim : bool
        If True, show this animation at the end of the function call
        (with a call to plt.show()).
    """

    # Attaching 3D axis to the figure
    fig = plt.figure(figName)
    ax = p3.Axes3D(fig)

    # Get data min-max values
    xMin = min([min(data[0, :]) for data in lData])
    xMax = max([max(data[0, :]) for data in lData])
    yMin = min([min(data[1, :]) for data in lData])
    yMax = max([max(data[1, :]) for data in lData])
    zMin = min([min(data[2, :]) for data in lData])
    zMax = max([max(data[2, :]) for data in lData])

    # Set axis label and boundary
    boundRatio = 1.1
    ax.set_xlim3d([xMin*boundRatio, xMax*boundRatio])
    ax.set_xlabel('X')
    ax.set_ylim3d([yMin*boundRatio, yMax*boundRatio])
    ax.set_ylabel('Y')
    ax.set_zlim3d([zMin*boundRatio, zMax*boundRatio])
    ax.set_zlabel('Z')

    # Select data that will be plot (depending on deltaData)
    trajData = [data[:, ::deltaData] for data in lData]
    trajTime = lTimes[::deltaData]
    nStep = len(trajTime)

    # Add fixed points if wanted
    if lFixedPoints:
        for coords in lFixedPoints:
            ax.scatter(coords[0], coords[1], coords[2], facecolors='k')

    timeTemplate = 'time={:1.2f}'
    timeText = ax.text2D(0.5, 0.95, timeTemplate.format(trajTime[0]),
                         transform=ax.transAxes)

    if wholeTraj:
        # Initial plot
        trajLines = [ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])[0]
                     for data in trajData]

        # Function used to update the plot
        def updateCurve(step):
            for line, data in zip(trajLines, trajData):
                line.set_data(data[0:2, :step])
                line.set_3d_properties(data[2, :step])
            timeText.set_text(timeTemplate.format(trajTime[step]))
            return trajLines

    else:
        # Style properties of the scatter plot
        lCol = plt.rcParams['axes.prop_cycle'].by_key()['color']
        lSym = ['o', 's', '^']
        lSize = [20, 100, 100]
        nCurveMax = min([len(l) for l in [lCol, lSym, lSize]])
        if len(trajData) > nCurveMax:
            raise ValueError(
                'Cannot plot more than {} trajectories'.format(nCurveMax))

        # Initial plot
        trajLines = [ax.scatter(data[0, 0], data[1, 0], data[2, 0], s=[sz],
                                marker=sy, facecolors='none', edgecolors=c)
                     for data, c, sy, sz in zip(trajData, lCol, lSym, lSize)]

        # Function used to update the plot
        def updateCurve(step):
            for line, data in zip(trajLines, trajData):
                line._offsets3d = ([data[0, step]],
                                   [data[1, step]],
                                   [data[2, step]])
            timeText.set_text(timeTemplate.format(trajTime[step]))
            return trajLines

    # Create the Animation object
    anim = animation.FuncAnimation(
        fig, updateCurve, nStep, interval=deltaFrame, blit=False)

    # Eventualy show animation
    if showAnim:
        plt.show()

    return anim
