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

#uniform (normalized) distribution on support [0,10]
def q(x):
    return 1/10.0+x-x

#target (approximately normalized) distribution on support [0,10]
def p(x):
    return (0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3))/1.03281

#return samples of p(x) distribution
def get_p_samples(n_samples):
    
    #uniform sample from q on support [0,10]
    q_samples = np.random.rand(n_samples)*10

    #compute weights and normalize
    w = p(q_samples)/q(q_samples)
    w /= sum(w)

    #construct cumulative distribution of weights
    cum_w = [w[0]] 
    for i in range(1,len(w)):
        cum_w.append(cum_w[i-1] + w[i])
    
    #set of random numbers with uniform distribution on [0,1]
    u = np.random.rand(n_samples)
    
    p_samples = np.zeros(n_samples)
    index = 0
    for i in range(n_samples):
        #count all u's which are below cum[i]
        indices = np.where(u<cum_w[i])
        #construct sample 
        p_samples[index:index+np.size(indices)] = q_samples[i]
        #increase index
        index += np.size(indices)
        #used up u's are not considered anymore
        u[indices] = 2 
        
    return p_samples

print("Sampling-Importance Resampling Algorithm")
I = integrate.quad(lambda x: q(x),0,10)
print("integrate q(x): {:.5f}".format(I[0]))
I = integrate.quad(lambda x: p(x),0,10)
print("integrate p(x): {:.5f}".format(I[0]))

ns=30000
t0=time.time()
samples = get_p_samples(ns)
t1=time.time()
print('\nGenerate',len(samples),"samples")
print("Time [sec]",t1-t0)

print('\nexpected value of p(x) by integration: {}'.format(integrate.quad(lambda x: x*p(x),0,10)[0]))
print('expected value of p(x) with samples from p(x): {}'.format(np.mean(samples)))
print('expected value of p(x) with uniform samples: {}'.format(10*np.mean(list(map(lambda x: x*p(x), (10*np.random.rand(ns)))))))



x = np.arange(0,4,0.01)
x2 = np.arange(0,4,0.01)
pl.plot(x,p(x),'k',lw=2)
pl.plot(x2,q(x2),'k--',lw=2)

pl.hist(samples,50, density =1,fc='k')
pl.xlabel('x',fontsize=10)
pl.ylabel('p(x)',fontsize=10)
pl.axis([0,4,0,1])
pl.show()


