#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:48:38 2018

@author: lunet
"""
import numpy as np


def dahlquistFE(lam, y0, T, nStep):
    dt = T/nStep
    g = 1+lam*dt
    p = np.arange(nStep+1)
    y = g**p
    y *= y0
    t = np.linspace(0, T, nStep+1)
    return t, y


def dahlquistBE(lam, y0, T, nStep):
    dt = T/nStep
    g = 1/(1-lam*dt)
    p = np.arange(nStep+1)
    y = g**p
    y *= y0
    t = np.linspace(0, T, nStep+1)
    return t, y
