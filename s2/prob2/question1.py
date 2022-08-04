#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:18:00 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt


from nievergelt import forwardEuler


def rhs(t, u):
    return np.cos(t)*u


def analytical(t):
    return np.exp(np.sin(t))


nStep = 500
t, u = forwardEuler(rhs, 0, 2*np.pi, 1, nStep)

plt.figure()
plt.plot(t, u.ravel(), label='Numerical')
plt.plot(t, analytical(t), 'o', markevery=0.1, label='Exact')
plt.legend()
plt.xlabel('$t$')
plt.tight_layout()


def computeError(nStep):
    t, uNum = forwardEuler(rhs, 0, 2*np.pi, 1, nStep)
    uExact = analytical(t)
    return np.linalg.norm((uNum.ravel()-uExact)/uExact, ord=np.inf)


lError = []
nSteps = [100, 500, 1000, 5000, 10000, 20000]
for nStep in nSteps:
    lError.append(computeError(nStep))

plt.figure()
plt.loglog(nSteps, lError, label='Maximum error')
plt.grid()
plt.legend()
plt.xlabel('$N_{step}$')
plt.tight_layout()
