#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:57:08 2018

@author: lunet
"""
import numpy as np
import matplotlib.pyplot as plt
from solver import finDiffMatrixD2C2

J = 14
L = 1

A = finDiffMatrixD2C2(J, L)
lamNum = np.linalg.eigvals(A)
plt.plot(lamNum.real, lamNum.imag, 'o', label='Numerical')

j = np.arange(J)+1
h = L/(J+1)
lamTh = -2/h**2 * (1+np.cos(np.pi*j/(J+1)))
plt.plot(lamTh.real, lamTh.imag, '+', label='Analytical')

plt.xlabel(r'$Re(\lambda)$')
plt.ylabel(r'$Im(\lambda)$')
plt.grid()
plt.legend()
