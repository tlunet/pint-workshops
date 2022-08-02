#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:35:52 2018

@author: lunet
"""
import numpy as np
import lorenz as lo


# Definition of the function used for time integration
def f(t, u):
    return lo.lorenzOperator(u[0], u[1], u[2])


# Local error computation
nStep = 1
nStepRef = 1000
dt = np.array([0.0001, 0.001, 0.01, 0.1])
lErrLoc = []
for T in dt:
    t, u = lo.forwardEuler(f, 0, T, [20, 5, -5], nStep)
    t, uRef = lo.forwardEuler(f, 0, T, [20, 5, -5], nStepRef)
    err = np.linalg.norm(uRef[:, -1] - u[:, -1])
    lErrLoc.append(err)

# Global error computation
T = max(dt)
lStep = np.array(T/dt, dtype=int)
nStepRef = 100*max(lStep)
t, uRef = lo.forwardEuler(f, 0, T, [20, 5, -5], nStepRef)
lErrGlob = []
for nStep in lStep:
    t, u = lo.forwardEuler(f, 0, T, [20, 5, -5], nStep)
    err = np.linalg.norm(uRef[:, -1] - u[:, -1])
    lErrGlob.append(err)

# Q-1(h): error plot
lo.plot2DCurve(dt, lErrLoc, 'Err', 'o--', label='Loc. trun. error')
lo.plot2DCurve(dt, lErrGlob, 'Err', 's--', label='Glob. trun. error')
lo.plot2DCurve(dt, 10**3.5 * dt, 'Err', label='Order 1')
lo.plot2DCurve(dt, 10**3 * dt**2, 'Err', label='Order 2',
               logX=True, logY=True, xLabel='$\\Delta_t$')
