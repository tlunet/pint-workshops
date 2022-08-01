#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:08:31 2018

Script to compute the eigenvalue for the Jacobian of the Lorenz System
"""
import numpy as np

# Set Lorenz system parameters
sigma, rho, beta = 10, 28, 8/3

K = (beta*(rho-1))**0.5

# Build the Jacobian matrix at one fixed point
mJac = np.array([[-sigma, sigma, 0],
                 [1, -1, -K],
                 [K, K, -beta]])

# Compute the eigenvalues
vLam = np.linalg.eigvals(mJac)
nLam = len(vLam)

# Print the eigenvalues
for i in range(nLam):
    lam = vLam[i]
    print('lam{} = {:1.2f} + {:1.2f}j'.format(i, np.real(lam), np.imag(lam)))
