#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Using the two-layer perceptron to solve a simple classification problem.
Author: Raoul Malm
"""

from pylab import *
import numpy as np  #numerical package for scientific computing
import TwoLayerPerceptron

def preprocessIris(infile,outfile):
    """ replace the plant names with numbers and generate new file """
    
    stext1 = 'Iris-setosa'
    stext2 = 'Iris-versicolor'
    stext3 = 'Iris-virginica'
    rtext1 = '0'
    rtext2 = '1'
    rtext3 = '2'
    
    fid = open(infile,"r")
    oid = open(outfile,"w")
    
    for s in fid:
        if s.find(stext1) > -1:
            oid.write(s.replace(stext1,rtext1))
        elif s.find(stext2) > -1:
            oid.write(s.replace(stext2,rtext2))
        elif s.find(stext3) > -1:
            oid.write(s.replace(stext3,rtext3))
    
    fid.close()
    oid.close()
    
#replace plant names with numbers in original file
#preprocessIris('iris.data','iris_proc.data')

#normalise data
iris = np.loadtxt('iris_proc.data',delimiter=',')
iris[:,:4] = iris[:,:4] - iris[:,:4].mean(axis=0)
imax = np.concatenate(((iris.max(axis=0)*np.ones((1,5))),(iris.min(axis=0)*np.ones((1,5)))),axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]

#specify target such that only one of three neurons fires
target = np.zeros((np.shape(iris)[0],3))
indices = np.where(iris[:,4]==0)
target[indices,0]=1
indices = np.where(iris[:,4]==1)
target[indices,1]=1
indices = np.where(iris[:,4]==2)
target[indices,2]=1

#randomise the order of the data
order = np.arange(np.shape(iris)[0])
np.random.shuffle(order)
iris = iris[order,:]
target = target[order,:]

#define train, valid, test data
train = iris[0::2,0:4]
traint = target[0::2]
valid = iris[1::4,0:4]
validt = target[1::4]
test = iris[3::4,0:4]
testt = target[3::4]

net = TwoLayerPerceptron.mlpcn(train,traint,4,0.2,'softmax','batch')

print('...perceptron training...')
(trainerror,validerror) = net.mlptrain_automatic(train,traint,valid,validt,100)
    
print()
print('Training stop after',len(trainerror)*100,'iterations')
print('Final Train Error',net.errfunc(net.mlpfwd(train,True)[1],traint))
print('Final Valid Error',net.errfunc(net.mlpfwd(valid,True)[1],validt))
print()

net.confmat(test,testt)
 
plot(np.arange(len(trainerror)),trainerror,'-b',label = 'train error')
plot(np.arange(len(validerror)),validerror,'-r',label = 'valid error')
legend(loc = 'upper right')
show()




















