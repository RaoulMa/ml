#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Using a two-layer network to predict the ozone layer thickness 
from data above Palmerston North in New Zealand between 1996 and 2004
Author: Raoul Malm
"""

from pylab import *
import numpy as np  #numerical package for scientific computing
import TwoLayerPerceptron

#ozone layer thickness above Palmerston North in New Zealand between 1996 and 2004
pnoz = loadtxt('data/PNoz.data')

#create [day,ozone] array
#inputs = np.concatenate((np.transpose(np.ones((1,np.shape(pnoz)[0]))*np.arange(np.shape(pnoz)[0])),np.transpose(np.ones((1,np.shape(pnoz)[0]))*pnoz[:,2])),axis=1)

#normalise data
pnoz[:,2] = pnoz[:,2]- pnoz[:,2].mean()
pnoz[:,2] = pnoz[:,2]/pnoz[:,2].max()

#assemble input vectors: x(a+t) = f(x(a),x(a-t),x(a-2t),...,x(a-kt))
t = 1 #stepsize
k = 4 #k points in the past used to predict the future point

lastPoint = np.shape(pnoz)[0]-t*(k+1)
inputs = np.zeros((lastPoint,k))
targets = np.zeros((lastPoint,1))

for i in range(lastPoint):
    inputs[i,:] = pnoz[i:i+t*k:t,2]
    targets[i] = pnoz[i+t*(k+1),2]

train = inputs[:-400:2,:]
traintarget = targets[:-400:2]
valid = inputs[1:-400:2,:]
validtarget = targets[1:-400:2]
test = inputs[-400:,:]
testtarget = targets[-400:]

"""
# randomly order the data
change = np.arange(np.shape(inputs)[0])
np.random.shuffle(change)
inputs = inputs[change,:]
targets = targets[change,:]
"""

#plot ozone versus days
xlabel('Days')
ylabel('normalized ozone')
plot(np.arange(0,2*np.shape(pnoz[:-400:2,2])[0],2),pnoz[:-400:2,2],'.r',label='train data')
plot(np.arange(1,2*np.shape(pnoz[1:-400:2,2])[0],2),pnoz[1:-400:2,2],'.g',label='valid data')
plot(np.arange(np.shape(pnoz[:-400,2])[0],np.shape(pnoz[:,2])[0],1),pnoz[-400:,2],'.b',label='test data')
legend(loc = 'upper right')
show()

net = TwoLayerPerceptron.mlpcn(train,traintarget,3,0.25,'linear','batch') 

trainerror = np.array([])
validerror = np.array([])

print('\nStart Train Error',net.errfunc(net.mlpfwd(train,True)[1],traintarget))
print('Start Valid Error',net.errfunc(net.mlpfwd(valid,True)[1],validtarget))

print('...perceptron training...')
(trainerror,validerror) = net.mlptrain_automatic(train,traintarget,valid,validtarget,100)

#for n in range(100):
#    trainerror = np.append(trainerror,net.errfunc(net.mlpfwd(train,True)[1],traintarget))
#    validerror = np.append(validerror,net.errfunc(net.mlpfwd(valid,True)[1],validtarget))
#    net.mlptrain(100)

print('Final Train Error',net.errfunc(net.mlpfwd(train,True)[1],traintarget))
print('Final Valid Error',net.errfunc(net.mlpfwd(valid,True)[1],validtarget))
  
plot(np.arange(len(trainerror)),trainerror,'-b',label = 'train error')
plot(np.arange(len(validerror)),validerror,'-r',label = 'valid error')
legend(loc = 'upper right')
show()

testout = net.mlpfwd(test,True)[1]

print('Test Error:',net.errfunc(testout,testtarget))
plot(np.arange(np.shape(test)[0]),testout,'.')
plot(np.arange(np.shape(test)[0]),testtarget,'x')
legend(('Predictions','Targets'))
show()






