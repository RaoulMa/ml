#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Choose a set of data points as weights and calculate RBF nodes for the
first layer. Those are then used as inputs for a one-layer perceptron, which gives the
output
"""

import numpy as np
import pcn

class rbf:
    """ radial basic function """
    
    def __init__(self,inputs,targets,nRBF,sigma=0,normalise=0,eta=0.25,functype='sigmoid',traintype='batch'):
        """ constructor """
        
        self.inputs = inputs
        self.targets = targets
        self.nRBF = nRBF #number of RBF nodes
        self.normalise = normalise
        self.eta = eta #learning rate
        self.functype = functype
        self.traintype = traintype
        
        #set width of gaussian
        if sigma==0:
            d = (self.inputs.max(axis=0)-self.inputs.min(axis=0)).max()
            self.sigma = d/np.sqrt(2*nRBF)  
        else:
            self.sigma = sigma
                
        #input array of RBF nodes
        self.hidden = np.zeros((np.shape(self.inputs)[0],self.nRBF))
        
        #set RBF weights to be random datapoints
        self.weights = np.zeros((np.shape(inputs)[1],self.nRBF))
        indices = np.arange(np.shape(self.inputs)[0])
        np.random.shuffle(indices)
        for i in range(self.nRBF):
            self.weights[:,i] = self.inputs[indices[i],:]
            
        #calculate the hidden rbf nodes (first layer)
        self.hidden = self.rbffwd(self.inputs,1)

        #use initialise perceptron for second layer
        self.perceptron = pcn.pcn(self.hidden,self.targets,self.eta,self.functype,self.traintype)

    def errfunc(self,outputs,targets):
        """ error function """
        E = 1/2*np.trace(np.dot(np.transpose(targets-outputs),targets-outputs))
        return E
            
    def rbftrain(self,nIt=100):
        """ training the network """
        #train perceptron
        self.perceptron.pcntrain(nIt)
        
    def rbftrain_automatic(self,valid,validt,itSteps):
        """ train the perceptron until the error on the validation data increases """
        
        #calculate the hidden rbf nodes (first layer)
        rbfvalid = self.rbffwd(valid,1)

        trainerror = np.array([])
        validerror = np.array([])

        (trainerror,validerror) = self.perceptron.pcntrain_automatic(rbfvalid,validt,itSteps)
        
        return trainerror,validerror
        
    def rbffwd(self,inputs,layer):
        """ run the network forward """
        
        #rbf nodes
        hidden = np.zeros((np.shape(inputs)[0],self.nRBF))

        #calculate gaussian overlap of input with weights
        for i in range(self.nRBF):
            hidden[:,i] = np.exp(-np.sum((inputs - np.ones((1,np.shape(inputs)[1]))*self.weights[:,i])**2,axis=1)/(2*self.sigma**2))

        #normalise RBF layer
        if self.normalise:
            hidden[:,:] /= np.transpose(np.ones((1,np.shape(hidden)[0]))*hidden[:,:].sum(axis=1))
        
        #output of hidden (rbf) layer 
        outputs = hidden
        
        #output of perceptron layer
        if layer == 2:
            outputs = self.perceptron.pcnfwd(hidden,True)
        
        return outputs

    def confmat(self,inputs,targets):
        """ confusion matrix to evaluate the performance of the network """
        
        #calculate hidden nodes
        hidden = self.rbffwd(inputs,1)
        
        #confusion matrix of perceptron
        self.perceptron.confmat(hidden,targets)
        
        return 0
 










