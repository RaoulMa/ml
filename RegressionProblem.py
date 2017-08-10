# -*- coding: utf-8 -*-
"""
Description: Using the two-layer perceptron to solve a simple regression problem.
Author: Raoul Malm
"""

from pylab import *
import numpy as np  #numerical python package for scientific computing
import TwoLayerPerceptron #class for two-layer perceptron

x = np.ones((1,40))*np.linspace(0,1,40)
t = np.sin(2*np.pi*x) + np.cos(4*np.pi*x)+np.random.randn(40)*0.2
x = np.transpose(x)
t = np.transpose(t)

train = x[0::2,:]
test = x[1::4,:]
valid = x[3::4,:]
traintarget = t[0::2,:]
testtarget = t[1::4,:]
validtarget = t[3::4,:]

p = TwoLayerPerceptron.mlpcn(train,traintarget,6,0.25,'linear','batch') 

trainerror = np.array([])
validerror = np.array([])

for n in range(200):
    trainerror = np.append(trainerror,p.errfunc(p.mlpfwd(train,True)[1],traintarget))
    validerror = np.append(validerror,p.errfunc(p.mlpfwd(valid,True)[1],validtarget))
    p.mlptrain(100)

print('\nFinal Train Error',p.errfunc(p.mlpfwd(train,True)[1],traintarget))
print('Final Valid Error',p.errfunc(p.mlpfwd(valid,True)[1],validtarget))
    
plot(np.arange(len(trainerror)),trainerror,'-b',label = 'train error')
plot(np.arange(len(validerror)),validerror,'-r',label = 'valid error')
legend(loc = 'upper right')
show()

plot(train,p.o2,'-b',label = 'train output')
plot(train,traintarget,'-r',label = 'train target')
legend(loc = 'upper right')
show()

plot(valid,p.mlpfwd(valid,True)[1],'-b',label = 'valid output')
plot(valid,validtarget,'-r',label = 'valid target')
legend(loc = 'upper right')
show()



