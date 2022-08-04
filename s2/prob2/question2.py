#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:49:33 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from nievergelt import forwardEuler, nievergelt

# Problem parameters
tBeg = 0
tEnd = 2*np.pi
u0 = 1

# PinT parameters
N = 10
nF = 1000
nG = 10

# Nievergelt parameters
Mn = 2
delta = 0.5

# Right Hand Side function
def rhs(t, u):
    return np.cos(t)*u

# Definition of the fine solver
def fineSolver(u0, tBeg, tEnd):
    t, u = forwardEuler(rhs, tBeg, tEnd, u0, nF)
    return u.ravel()

# Definition of the coarse solver
def coarseSolver(u0, tBeg, tEnd):
    t, u = forwardEuler(rhs, tBeg, tEnd, u0, nG)
    return u.ravel()

# Fine solution (for plot reference)
tFinePlot, uFinePlot = forwardEuler(rhs, tBeg, tEnd, u0, nF*N)

# Run Nievergelt algorithm
uNiev, maxErr, uCoarse, uFine, shoot, listP, times = \
    nievergelt(u0, fineSolver, coarseSolver, tBeg, tEnd, N, Mn, delta)

# Plot results
plt.figure()
# -- continuous fine solution
plt.plot(tFinePlot, uFinePlot.ravel())
# -- fine solution on sub-interval bounds
plt.plot(times, uFine, 'o', label='Fine')
# -- coarse solution
plt.plot(times, uCoarse, 's', label='Coarse')
# -- shoot solutions
for i, interval in enumerate(shoot):
    for sol in interval:
        plt.plot(tFinePlot[i*nF:(i+1)*nF+1], sol["inner"], '--', c="gray")
# -- Nievergelt solution
plt.plot(times, uNiev, '^', label='Nievergelt')
# -- plot details
plt.legend()
plt.xlabel('Time'), plt.ylabel('Solution')
plt.tight_layout()
