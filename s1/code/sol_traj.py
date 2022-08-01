#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 00:49:11 2018

@author: telu
"""
import lorenz as lo
from math import log10

# Numerical settings
T = 20
nStep = 20000

# Parameters and position of fixed points
sigma, rho, beta = 10, 28, 8/3
xf1, zf1 = (beta*(rho-1))**0.5, rho-1
xf2, zf2 = -(beta*(rho-1))**0.5, rho-1


# Definition of the function used for time integration
def f(t, u):
    return lo.lorenzOperator(u[0], u[1], u[2], sigma, rho, beta)


# Baseling solve
t1, u1 = lo.forwardEuler(f, 0, T, [20, 5, -5], nStep)

# Small variation on the initial solution
epsilon1 = 1e-3
t2, u2 = lo.forwardEuler(f, 0, T, [20+epsilon1, 5, -5], nStep)

# Smaller variation on the initial solution
epsilon2 = 1e-10
t3, u3 = lo.forwardEuler(f, 0, T, [20+epsilon2, 5, -5], nStep)

# %% Q-1(f): Displaying animation
anim1 = lo.plotAnimated3DCurve(
    [u1], t1, 'anim1', wholeTraj=True, deltaData=10,
    lFixedPoints=[(xf1, xf1, zf1), (xf2, xf2, zf2)],
    deltaFrame=10, showAnim=True)

# %% Q-1(g): Displaying animation of the two trajectories
anim2 = lo.plotAnimated3DCurve(
    [u1, u2], t1, 'anim2', wholeTraj=False, deltaData=10,
    lFixedPoints=[(xf1, xf1, zf1), (xf2, xf2, zf2)],
    deltaFrame=10, showAnim=True)

# %% Q-1(g): Displaying the two x-trajectories
lo.plot2DCurve(
    t1, u1[0, :], 'x-traj', label='Original')
lo.plot2DCurve(
    t2, u2[0, :], 'x-traj', label='{:1.0e} variation'.format(epsilon1))
lo.plot2DCurve(
    t3, u3[0, :], 'x-traj', label='{:1.0e} variation'.format(epsilon2),
    xLabel='Time')

# %% Q-1(g): Displaying the difference between x-trajectories
lo.plot2DCurve(
    t1, abs(u1[0, :]-u2[0, :]), 'x-diff-traj',
    label='$\\epsilon=10^{'+str(int(log10(epsilon1)))+'}$')
lo.plot2DCurve(
    t1, abs(u1[0, :]-u3[0, :]), 'x-diff-traj', logY=True, xLabel='Time',
    label='$\\epsilon=10^{'+str(int(log10(epsilon2)))+'}$')
