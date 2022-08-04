#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:06:14 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from parareal import parareal, backwardEulerLin, finDiffMatrixD2C2

# Problem parameters
J = 99
L = 1
u0 = 20*np.ones(J)
T = 1/20

# PinT parameters
N = 10

# Parareal parameters
nF = 100
nG = 1
K = 6

# Differenciation matrix (RHS)
A = finDiffMatrixD2C2(J, L)

# Source term function
def source(t):
    return 0

# Definition of the fine solver
def fineSolver(u0, tBeg, tEnd):
    t, u = backwardEulerLin(A, u0, source, tBeg, tEnd, nF)
    return u[:, -1]

# Definition of the coarse solver
def coarseSolver(u0, tBeg, tEnd):
    t, u = backwardEulerLin(A, u0, source, tBeg, tEnd, nG)
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

# %% Function to compute Parareal efficiency

def computeEfficiency(eps, nG):

    def coarseSolver(u0, tBeg, tEnd):
        t, u = backwardEulerLin(A, u0, source, tBeg, tEnd, nG)
        return u[:, -1]

    Uk, uF, err, times = parareal(u0, 0, T, N,
        fineSolver, coarseSolver, K)

    kMin = np.argmax(err < eps)
    efficiency = 1/((N+kMin)*nG/nF + kMin)

    return efficiency, kMin

eps = 1e-3
nGVals = np.arange(20)+1
effVals = []
kMinVals = []
for nG in nGVals:
    eff, kMin = computeEfficiency(eps, nG)
    effVals.append(eff)
    kMinVals.append(kMin)

plt.figure('Efficiency')
plt.plot(nGVals, effVals, 'o-')
plt.xlabel('')
