#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Log of Share price = Drift plus Random Walk
Author: Raoul Malm
"""

import pylab as pl
import numpy as np

mu = 0.06 #drift = growth rate per year
sig = 0.18 #volatility
n = 100  #years
ns = 1000 #number of steps
dt = n/ns #step size

S = np.zeros(ns)
S[0] = 1

for i in range(ns-1):
    S[i+1] = S[i]*np.exp(mu*dt+sig*np.random.normal(0,np.sqrt(dt)))    

x = np.arange(0,ns,1)
pl.plot(dt*x,S[x],color='b',lw=1,label='mu=' + str(mu) + ', sig=' + str(sig))
pl.plot(dt*x,S[0]*np.exp(mu*dt*x),color='black',label='drift')
pl.legend(loc = 'upper left')
pl.xlabel('100 years')
pl.ylabel('S(t)')
pl.show()
