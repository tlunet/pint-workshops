#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:45:09 2022

@author: telu
"""
from parareal import parareal, lorenzOperator, forwardEuler
import numpy as np
import matplotlib.pyplot as plt

# ... in progress

# Problem parameters
u0 = [20, 5, -5]
T = 5

# PinT parameters
N = 10

# Parareal parameters
nF = 100
nG = 1
K = 6

# Definition of the function used for time integration
def f(t, u):
    return lorenzOperator(u[0], u[1], u[2])

# %% Accuracy for Lorentz system
nStep = 200000

t1, u1 = forwardEuler(f, 0, T, u0, nStep)
t2, u2 = forwardEuler(f, 0, T, u0, nStep*2)


# %%
# Definition of the fine solver
def fineSolver(u0, tBeg, tEnd):
    t, u = forwardEuler(f, tBeg, tEnd, u0, nF)
    return u[:, -1]

# Definition of the coarse solver
def coarseSolver(u0, tBeg, tEnd):
    t, u = forwardEuler(f, tBeg, tEnd, u0, nG)
    return u[:, -1]

# Parareal solution
Uk, uF, err, times = parareal(u0, 0, T, N,
    fineSolver, coarseSolver, K)

# %% Plot solution at T=1/20
x = np.linspace(0, L, num=J+2)[1:-1]
plt.figure('Solution')
iTime = -1
plt.plot(x, uF[iTime], '--o', c='gray', label='$u^F$', markevery=0.1)
plt.plot(x, Uk[0][iTime], '--s', c='brown', label='$u^G$', markevery=0.1)
for k in range(4):
    plt.plot(x, Uk[k+1][iTime], label='Parareal, $k={}$'.format(k+1))
plt.legend()
plt.grid(True)
plt.xlabel('$x$')

# %% Plot error for nG = 1
k = np.arange(K+1)
plt.figure('Error')
plt.semilogy(k, err, '-o', label='$n_G=1$')

# compute solution for nG = 2
nG = 2
def coarseSolver(u0, tBeg, tEnd):
    t, u = backwardEulerLin(A, u0, source, tBeg, tEnd, nG)
    return u[:, -1]
_, _, err2, _ = parareal(u0, 0, T, N, fineSolver, coarseSolver, K)

plt.semilogy(k, err2, '-o', label='$n_G=2$')
plt.hlines(1e-3, 0, K, colors='gray', linestyles='--')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.grid(True)
