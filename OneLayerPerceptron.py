# -*- coding: utf-8 -*-
"""
Description: Implementation of a one-dimensional perceptron with one neuron layer.
Author: Raoul Malm
"""
import numpy as np  #numerical python package for scientific computing

class pcn: 
    """ class for a basic perceptron """
    # input = N1xN2 matrix -> add bias node -> N1x(N2+1) matrix
    # weight = (N2+1)xN3 matrix
    # target = N1xN3 matrix
    # activation = input x weight
    
    def __init__(self,inputs,targets,eta):
        """ constructor """
        #add -1 bias node for input
        self.inputs = np.concatenate((-np.ones((len(inputs),1)),inputs),axis=1) 
        self.targets = targets
        self.weights = (np.random.rand(np.shape(self.inputs)[1],np.shape(self.targets)[1])*2-1)/np.sqrt(np.shape(self.inputs[1])) 
        self.eta = eta
        print('One-layer perceptron')
    
    def errfunc(self,outputs,targets):
        """ error function """
        E = 1/2*np.trace(np.dot(np.transpose(targets-outputs),targets-outputs))
        return E
    
    def pcntrain(self,nIt):
        """ train the perceptron """
        #for each number of iterations
        for n in range(nIt):
            #construct output array
            self.activations = np.zeros(np.shape(self.targets))
            #randomize ordering of inputs to learn more efficiently
            change = np.arange(len(self.inputs))
            np.random.shuffle(change)
            #for each number of inputs
            for i in set(change):
                #calculate neuron activations
                self.activations[i] = self.pcnfwd(self.inputs[i],self.weights)
                #adjust weights
                self.weights += self.eta*np.outer(np.transpose(self.inputs[i]),self.targets[i]-self.activations[i])  
                #print(n,'weights\n',self.weights)
            #calculate neuron activations
            self.activations = self.pcnfwd(self.inputs,self.weights)
            print(n,'Error',self.errfunc(self.activations,self.targets))
            #break loop in case targets are fulfilled
            if (self.activations==self.targets).all(): break
        return 0

    def pcnfwd(self,inputs,weights):
        """ run the network forward """
        activations =  np.dot(inputs,weights)
        #return neuron activations
        return np.where(activations>0,1,0)
    
    def confmat(self,inputs,targets):
        """ confusion matrix to evaluate the performance of the network """
        inputs = np.concatenate((-np.ones((len(inputs),1)),inputs),axis=1) 
        outputs = self.pcnfwd(inputs,self.weights)	
        classes = np.unique(targets,axis=0)
        nClasses = np.shape(classes)[0]
        cm = np.zeros((nClasses,nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                for n in range(np.shape(targets)[0]):
                    if (outputs[n]==classes[i]).all() and (targets[n]==classes[j]).all():
                        cm[i,j]+=1
        #print('targets\n',targets)
        #print('outputs\n',outputs)
        #print('classes\n',classes)        
        print('confusion matrix\n',cm)
        if np.sum(cm)!=0:
            print('succeed rate',100*np.trace(cm)/np.sum(cm),'%')
        return 0
            



