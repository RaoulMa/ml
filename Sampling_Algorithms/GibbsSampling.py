#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Gibbs sampling algorithm
Author: Raoul Malm
"""

import pylab as pl
import numpy as np
import scipy.integrate as integrate


#Gaussian expectation values and standard deviations
mx = 10.
my = 20.
sx = 2.
sy = 3.

#p(x)
def px(x):
    return np.exp(-(x-mx)**2/(2.*sx**2)*(1-1/(sx**2*sy**2)))

#p(y)
def py(y):
    return np.exp(-(y-my)**2/(2*sy**2)*(1-1/(sx**2*sy**2)))

#p(x|y)
def pxgiveny(x,y):
    return np.exp(-(x-mx-(y-my)/sy**2)**2/(2*sx**2))

#p(y|x)
def pygivenx(y,x):
    return np.exp(-(y-my-(x-mx)/sx**2)**2/(2*sy**2))

#p(x,y)
def pxy(x,y):
    return np.exp(-((x-mx)**2/(2*sx**2) - (x-mx)*(y-my)/(sx**2*sy**2) + (y-my)**2/(2*sy**2)))    
    
#get x sample from p(x|y)
def xgiveny(y):
    return np.random.normal(mx + (y-my)/sy**2,sx)

#get y sample from p(y|x)
def ygivenx(x):
    return np.random.normal(my + (x-mx)/sx**2,sy)

#check that p(x,y) = p(x|y) p(y) and p(x,y) = p(y|x) p(x)

#number of samples
ns=1000 

#Gibbs sampling algorithm
def gibbs():
    x0 = np.zeros(ns,dtype=float)
    y0 = np.zeros(ns,dtype=float)
    
    for i in range(ns):
        y = np.random.rand(1)
        for j in range(2):
            x = xgiveny(y)
            y = ygivenx(x)
        x0[i] = x
        y0[i] = y
        
    
    return x0,y0

#start the Gibbs sampling
samples = gibbs()

print('Gibbs sampling algorithm with p(x,y)')
I = integrate.quad(lambda x: px(x),0,20)
print("p(x) normal.:",I)
I = integrate.quad(lambda x: py(x),0,20)
print("p(y) normal.:",I)

pl.figure(1)
x = np.arange(0,20,1)
pl.hist(samples[0],bins=x,fc='k',normed=1)
x = np.arange(0,20,0.1)
pl.plot(x,10*px(x)/np.sum(px(x)),color='b',lw=3)
pl.title('MCMC with Gibbs sampling')
pl.xlabel('x')
pl.ylabel('p(x)')

pl.figure(2)
y = np.arange(0,40,1)
pl.hist(samples[1],bins=y,fc='k',normed=1)
y = np.arange(0,40,0.1)
pl.plot(y,10*py(y)/np.sum(py(y)),color='b',lw=3)
pl.title('MCMC with Gibbs sampling')
pl.xlabel('y')
pl.ylabel('p(y)')

pl.show()
