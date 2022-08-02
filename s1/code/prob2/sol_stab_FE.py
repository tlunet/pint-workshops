#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:44:18 2018

@author: lunet
"""
import numpy as np
import matplotlib.pyplot as plt

# Plot stability region of Forward Euler
theta = np.linspace(0, 2*np.pi, num=10000)
circle = -1+np.exp(1j*theta)
plt.plot(circle.real, circle.imag)

# Lambda with their associated dtMax
vLam = [-1, 1j, 1j-2]
vDtMax = [2, 1.1, 4./5]

# Plot each curve
for lam, dtMax in zip(vLam, vDtMax):
    dt = np.linspace(0, dtMax)
    r = dt*lam
    plt.plot(r.real, r.imag, label='$\\lambda=$'+str(lam))

# Axis general settings
plt.legend()
plt.axis('scaled')
plt.grid(True)
