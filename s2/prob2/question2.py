#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:49:33 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from nievergelt import forwardEuler


tBeg = 0
tEnd = 2*np.pi
u0 = 1

N = 10
nF = 1000
nG = 10


def rhs(t, u):
    return np.cos(t)*u


def fineSolver(u0, tBeg, tEnd):
    t, u = forwardEuler(rhs, tBeg, tEnd, u0, nF)
    return u.ravel()


def coarseSolver(u0, tBeg, tEnd):
    t, u = forwardEuler(rhs, tBeg, tEnd, u0, nG)
    return u.ravel()


times = np.linspace(tBeg, tEnd, num=N+1)
plt.figure()



# Fine solution (for reference)
tFinePlot, uFinePlot = forwardEuler(rhs, tBeg, tEnd, u0, nF*N)
plt.plot(tFinePlot, uFinePlot.ravel())

uFine = [u0 for _ in range(N+1)]
for i in range(N):
    uFine[i+1] = fineSolver(uFine[i], times[i], times[i+1])[-1]
plt.plot(times, uFine, 'o', label='Fine')


# Coarse propagation
uCoarse = [u0 for _ in range(N+1)]
for i in range(N):
    uCoarse[i+1] = coarseSolver(uCoarse[i], times[i], times[i+1])[-1]
plt.plot(times, uCoarse, 's', label='Coarse')


# Shooting solutions
shoot = [[] for _ in range(N)]
# -- first shoot is fine solve
uLeft = uCoarse[0]
uInner = fineSolver(u0, times[0], times[1])
uRight = uInner[-1]
shoot[0].append({"left": uLeft, "right": uRight, "inner": uInner})
# -- multiple shooting (n=2)
offset = 0.2
for i in range(1, N):
    # First shoot
    uLeft = uCoarse[i] + offset
    uInner = fineSolver(uLeft, times[i], times[i+1])
    uRight = uInner[-1]
    shoot[i].append({"left": uLeft, "right": uRight, "inner": uInner})
    # Second shoot
    uLeft = uCoarse[i] - offset
    uInner = fineSolver(uLeft, times[i], times[i+1])
    uRight = uInner[-1]
    shoot[i].append({"left": uLeft, "right": uRight, "inner": uInner})

for i, interval in enumerate(shoot):
    for sol in interval:
        plt.plot(tFinePlot[i*nF:(i+1)*nF+1], sol["inner"], '--', c="gray")


uNiev = [u0 for _ in range(N+1)]
# uNiev[0] = shoot[0][0]["left"] --- implicit
uNiev[1] = shoot[0][0]["right"]
# Linear interpolation
listP = []
for i in range(1, N):
    u1 = uNiev[i]
    uS1 = shoot[i][0]["left"]
    uS2 = shoot[i][1]["left"]
    p = (u1 - uS2)/(uS1 - uS2)
    listP.append(p)
    uNiev[i+1] = p * shoot[i][0]["right"] + (1 - p) * shoot[i][1]["right"]

plt.plot(times, uNiev, '^', label='Nievergelt')

plt.legend()
plt.xlabel('Time'), plt.ylabel('Solution')
plt.tight_layout()
