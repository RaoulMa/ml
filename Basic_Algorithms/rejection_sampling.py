#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Basic rejection sampling algorithm.
Author: Raoul Malm
"""

import pylab as pl
import numpy as np
import time

#uniform normalized distribution on support [0,4]
def q(x):
    return 1/4*(1+x-x);

#return samples according to uniform distribution q(x) on support [0,4]
def xq():
    return np.random.rand()*4.

#target distribution (normalized)
#maximum = 0.716
def p(x):
    return 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3) 

#return samples according to p(x) distribution
def xp(ns,M):    
    samples = np.zeros(ns,dtype=float) #ns samples
    count = 0 #count all points required to find ns samples
    for i in range(ns):
        accept = False
        while not accept:
            x = xq() #sample from q(x) distribution
            u = np.random.rand()*q(x)*M #u from uniform [0,Mq(x)] distribution
            if u<p(x):
                accept = True
                samples[i] = x
            else: 
                count += 1
    return samples,count

M=0.8*4
ns=10000
t0=time.time()
(samples,count) = xp(ns,M)
t1=time.time()
print("Basic rejecting sampling algorithm")
print("Time [sec]",t1-t0)
print(len(samples),"Samples from",count,"Points")

x = np.arange(0,4,0.01)
x2 = np.arange(0,4,0.01)
pl.plot(x,p(x),'k',lw=2)
pl.plot(x2,q(x)*M,'k--',lw=2)

pl.hist(samples,40,normed=1,fc='k')
pl.xlabel('x',fontsize=10)
pl.ylabel('p(x)',fontsize=10)
pl.axis([0,4,0,1])
pl.show()
