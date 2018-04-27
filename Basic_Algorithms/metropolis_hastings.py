#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: The Metropolis-Hastings algorithm. This is a Markov Chain Monte Carlo
sampling alogorithm.
Author: Raoul Malm
"""

# 
import pylab as pl
import numpy as np
import scipy.integrate as integrate

#target distribution
def p(x):
    mu1 = 3 #expectation value1
    mu2 = 10 #expectation value2
    v1 = 10 #variance1
    v2 = 3 #variance2
    return 0.3*np.exp(-(x-mu1)**2/v1) + 0.7*np.exp(-(x-mu2)**2/v2)

#proposal distribution = normalized Gaussian
mu = 5
sigma = 5
def q(x):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x-mu)**2/(2*sigma**2))

print("Metropolis-Hastings Algorithm")
I = integrate.quad(lambda x: q(x),-10,20)
print("q(x) normal.:",I)
I = integrate.quad(lambda x: p(x),-10,20)
print("p(x) normal.:",I)

N = 10000 #number of samples to be generated

#use fixed Gaussian as proposal distribution
#note: proposal distribution q(y) is not conditional
u = np.random.rand(N)
y = np.zeros(N)
y[0] = np.random.normal(mu,sigma)
for i in range(N-1):
    ynew = np.random.normal(mu,sigma)
    alpha = min(1,p(ynew)*q(y[i])/(p(y[i])*q(ynew))) #acceptance
    if u[i] < alpha:
        y[i+1] = ynew
    else:
        y[i+1] = y[i]

#use Gaussian with changing expectation value as proposal distribution
#note: this proposal distribution q(y1,y2) is symmetric in its arguments
u2 = np.random.rand(N)
y2 = np.zeros(N)
y2[0] = np.random.normal(0,sigma)
for i in range(N-1):
    y2new = y2[i] + np.random.normal(0,sigma)
    alpha = min(1,p(y2new)/p(y2[i])) #acceptance
    if u2[i] < alpha:
        y2[i+1] = y2new
    else:
        y2[i+1] = y2[i]

x = np.arange(-10,20,0.5)

pl.figure(1)
nbins = 30
pl.hist(y, bins = x,normed=1)
pl.plot(x, 1/3.8*p(x), color='black',lw=2)
pl.title('MCMC with fixed proposal distribution')
pl.xlabel('x',fontsize=15)
pl.ylabel('p(x)',fontsize=15)
pl.plot(x,q(x),'b--',lw=2)

pl.figure(2)
nbins = 30
pl.hist(y2, bins = x,normed=1)
pl.plot(x, 1/3.8*p(x), color='black',lw=2)
pl.title('MCMC with moving proposal distribution')
pl.xlabel('x',fontsize=15)
pl.ylabel('p(x)',fontsize=15)
pl.plot(x,q(x),'b--',lw=2)

pl.show()