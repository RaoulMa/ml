#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Basic sampling-importance-resampling algorithm.
Author: Raoul Malm
"""
import pylab as pl
import numpy as np
import time
import scipy.integrate as integrate

#uniform (normalized) distribution on support [0,4]
def q(x):
    return 1/4.0+x-x

#target (approximately normalized) distribution on support [0,4]
def p(x):
    return 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3) 

#return samples of p(x) distribution
def xp(ns):
    
    sample1 = np.zeros(ns)
    w = np.zeros(ns)
    sample2 = np.zeros(ns)
    
    #uniform sample from q on support [0,4]
    sample1 = np.random.rand(ns)*4

    #compute weights and normalize
    w = p(sample1)/q(sample1)
    w /= np.sum(w)

    #construct cumulative distribution of weights
    cumw = np.zeros(len(w))
    cumw[0] = w[0]
    for i in range(1,len(w)):
        cumw[i] = cumw[i-1]+w[i]
    
    #set of random numbers  with uniform distribution on [0,1]
    u = np.random.rand(ns)
    
    index = 0
    for i in range(ns):
        #count all u's which are below cum[i]
        indices = np.where(u<cumw[i])
        #construct sample 
        sample2[index:index+np.size(indices)] = sample1[i]
        #increase index
        index += np.size(indices)
        #used up u's are not considered anymore
        u[indices] = 2 
        
    return sample2

print("Sampling-Importance Resampling Algorithm")
I = integrate.quad(lambda x: q(x),0,4)
print("q(x) normal.:",I)
I = integrate.quad(lambda x: p(x),0,4)
print("p(x) normal.:",I)

ns=10000
t0=time.time()
samples = xp(ns)
t1=time.time()
print(len(samples),"Samples")
print("Time [sec]",t1-t0)

x = np.arange(0,4,0.01)
x2 = np.arange(0,4,0.01)
pl.plot(x,p(x),'k',lw=2)
pl.plot(x2,q(x2),'k--',lw=2)

pl.hist(samples,20,normed=1,fc='k')
pl.xlabel('x',fontsize=10)
pl.ylabel('p(x)',fontsize=10)
pl.axis([0,4,0,1])
pl.show()


