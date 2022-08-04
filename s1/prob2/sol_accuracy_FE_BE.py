#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:52:43 2018

@author: lunet
"""
import numpy as np
import matplotlib.pyplot as plt
from solver import dahlquistFE, dahlquistBE

T = 1
y0 = 1
nStep = 10

lC = plt.rcParams['axes.prop_cycle'].by_key()['color']

labelTh = 'Theorique'
labelFE = 'Forward Euler'
labelBE = 'Backward Euler'
for lam, s in zip([1j, 1j-1], ['-', '--']):
    t, yFE = dahlquistFE(lam, y0, T, nStep)
    t, yBE = dahlquistBE(lam, y0, T, nStep)
    yTh = np.exp(lam*t)

    plt.figure('traj')
    plt.plot(yTh.real, yTh.imag, 'o'+s, c=lC[0], label=labelTh)
    plt.plot(yFE.real, yFE.imag, 's'+s, c=lC[1], label=labelFE)
    plt.plot(yBE.real, yBE.imag, '^'+s, c=lC[2], label=labelBE)

    plt.figure('vabs')
    plt.plot(t, np.abs(yTh), 'o'+s, c=lC[0], label=labelTh)
    plt.plot(t, np.abs(yFE), 's'+s, c=lC[1], label=labelFE)
    plt.plot(t, np.abs(yBE), '^'+s, c=lC[2], label=labelBE)

    plt.figure('angle')
    plt.plot(t, np.angle(yTh), 'o'+s, c=lC[0], label=labelTh)
    plt.plot(t, np.angle(yFE), 's'+s, c=lC[1], label=labelFE)
    plt.plot(t, np.angle(yBE), '^'+s, c=lC[2], label=labelBE)

    labelTh, labelFE, labelBE = None, None, None

for figName in ['traj', 'vabs', 'angle']:
    plt.figure(figName)
    plt.grid()
    plt.legend()

plt.figure('traj')
plt.axis('scaled')
plt.xlim(-0.1, 1.1)
plt.xlabel('$Re(y)$')
plt.ylabel('$Im(y)$')
plt.savefig('dahlquist-traj.pdf', bbox_inches='tight')

for figName in ['vabs', 'angle']:
    plt.figure(figName)
    plt.xlabel('Time')
    plt.savefig('dahlquist-{}.pdf'.format(figName), bbox_inches='tight')
