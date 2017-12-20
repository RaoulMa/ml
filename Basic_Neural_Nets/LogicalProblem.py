# -*- coding: utf-8 -*-
"""
Description: Using different networks to solve basic logical problems. 
Author: Raoul Malm
"""

import numpy as np  #numerical python package for scientific computing
import pcn #class for one-layer perceptron
import mlpcn #class for two-layer perceptron
import rbf #class for radial basic function network

"""
#logical AND
inputs = np.array([[1,1],[1,0],[0,1],[0,0]]) 
targets = np.array([[1],[0],[0],[0]])
"""

"""
#logical OR
inputs = np.array([[1,1],[1,0],[0,1],[0,0]]) 
targets = np.array([[1],[1],[1],[0]])
"""

#logical XOR
inputs = np.array([[1,1],[1,0],[0,1],[0,0]]) 
targets = np.array([[0],[1],[1],[0]])

"""
#identity matrix
inputs = np.array([[1,1],[1,0],[0,1],[0,0]]) 
targets = np.array([[1,1],[1,0],[0,1],[0,0]])
"""

#use one-layer perceptron
p = pcn.pcn(inputs,targets,0.2,'linear','batch') 
p.pcntrain(10000)
p.confmat(inputs,targets)

#use rbf network
p = rbf.rbf(inputs,targets,4,0,1,0.2,'linear','batch') 
p.rbftrain(10000)
p.confmat(inputs,targets)

#use two-layer perceptron
p = mlpcn.mlpcn(inputs,targets,4,0.2,'linear','batch') 
p.mlptrain(10000)
p.confmat(inputs,targets)
