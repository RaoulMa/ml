# -*- coding: utf-8 -*-
"""
description: Train basic perceptrons for basic classification problems. 
author: Raoul Malm
"""

import numpy as np  #numerical python package for scientific computing
import OneLayerPerceptron #class for one-layer perceptron
import TwoLayerPerceptron #class for two-layer perceptron


#logical AND
inputs = np.array([[1,1],[1,0],[0,1],[0,0]]) 
targets = np.array([[1],[0],[0],[0]])

"""
#logical OR
inputs = np.array([[1,1],[1,0],[0,1],[0,0]]) 
targets = np.array([[1],[1],[1],[0]])
"""

"""
#logical XOR
inputs = np.array([[1,1],[1,0],[0,1],[0,0]]) 
targets = np.array([[0],[1],[1],[0]])
"""

"""
#identity matrix
inputs = np.array([[1,1],[1,0],[0,1],[0,0]]) 
targets = np.array([[1,1],[1,0],[0,1],[0,0]])
"""

"""
#use one-layer perceptron
p = pcn(inputs,targets,0.2) 
p.pcntrain(20)
p.confmat(inputs,targets)
"""

#use two-layer perceptron
p = mlpcn(inputs,targets,3,0.2,'sigmoid') 
p.pcntrain(1000)
p.confmat(inputs,targets)

