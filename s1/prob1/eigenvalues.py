#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:08:31 2018

Script to compute and display the eigenvalues of a matrix with given
coefficients, using the numpy package.
"""
import numpy as np

# Set some coefficients, a=21, b=12, ...
a, b, c = 31, 12, -27

# Build the matrix
mJac = np.array([[a, b, 0],
                 [a, 0, c],
                 [c, b, a]])

# Compute the eigenvalues
vLam = np.linalg.eigvals(mJac)

# Get the number of eigenvalues
nLam = len(vLam)

# Print the eigenvalues
for i in range(nLam):
    lam = vLam[i]
    print('lam{} = {:1.2f} + {:1.2f}j'.format(i, np.real(lam), np.imag(lam)))
