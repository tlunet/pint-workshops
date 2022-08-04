#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:12:16 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from nievergelt import forwardEuler


def rhs(t, u):
    return np.cos(t)*np.exp(-u)


def analytical(t):
    return np.log(np.sin(t)+2)


nStep = 500
tBeg = 0
tEnd = 2*np.pi
u0 = np.log(2)

t, u = forwardEuler(rhs, tBeg, tEnd, u0, nStep)

plt.figure()
plt.plot(t, u.ravel(), label='Numerical')
plt.plot(t, analytical(t), 'o', markevery=0.05, label='Exact')
plt.legend()
plt.xlabel('$t$')
plt.tight_layout()


# %%

def computeError(nStep):
    t, uNum = forwardEuler(rhs, tBeg, tEnd, u0, nStep)
    uExact = analytical(t)
    return np.linalg.norm(uNum.ravel()-uExact, ord=np.inf)


lError = []
nSteps = [10, 50, 100, 500, 1000, 5000, 10000, 20000]
for nStep in nSteps:
    lError.append(computeError(nStep))

plt.figure()
plt.loglog(nSteps, lError, label='Maximum error')
plt.grid()
plt.legend()
plt.xlabel('$N_{step}$')
plt.tight_layout()
