# -*- coding: utf-8 -*-
"""
Description: Implementation of a one-dimensional perceptron with one neuron 
layer. The user can specifiy the training method, the learning rate and the 
function type.
"""
import numpy as np  #numerical python package for scientific computing

class pcn: 
    """ class for a basic perceptron """
    # input = N1xN2 matrix -> add bias node -> N1x(N2+1) matrix
    # weight = (N2+1)xN3 matrix
    # target = N1xN3 matrix
    # activation = input x weight
    
    def __init__(self,inputs,targets,eta=0.25,functype='sigmoid',traintype='batch'):
        """ constructor """
        #add -1 bias node for input
        self.inputs = np.concatenate((-np.ones((len(inputs),1)),inputs),axis=1) 
        self.targets = targets
        self.weights = (np.random.rand(np.shape(self.inputs)[1],np.shape(self.targets)[1])*2-1)/np.sqrt(np.shape(self.inputs[1])) 
        self.eta = eta
        self.functype = functype
        self.traintype = traintype
        
        #print('One-layer perceptron')
        #print('with',np.shape(self.targets)[1],'output neurons')
        #print('with',np.shape(self.inputs)[0],'inputs / targets of dim',np.shape(self.inputs)[1]-1,'/',np.shape(self.targets)[1])
        #print('with',self.functype,'function for output neurons')
        #print('with learning rate',self.eta,'and',self.traintype,'training')
        
    
    def errfunc(self,outputs,targets):
        """ error function """
        E = 1/2*np.trace(np.dot(np.transpose(targets-outputs),targets-outputs))
        return E
    
    def pcntrain(self,nIt):
        """ train the perceptron """
        
        #construct output array
        self.outputs = self.pcnfwd(self.inputs,False)
        #print('Start Error',self.errfunc(self.outputs,self.targets))
        
        #for each number of iterations
        for n in range(nIt):
            
            if np.mod(n,1000)==0 and n!=0:
                print(n,'Error',self.errfunc(self.outputs,self.targets))
            
            #batch training 
            if self.traintype == 'batch':
                self.outputs = self.pcnfwd(self.inputs,False)
                #dwik = eta*xi*(tk-ok)
                self.weights += self.eta*np.dot(np.transpose(self.inputs),self.targets-self.outputs)  
            
            #sequential training
            elif self.traintype == 'sequential':
                #randomize ordering of inputs to learn more efficiently
                change = np.arange(len(self.inputs))
                np.random.shuffle(change)
                #for each number of inputs
                for i in set(change):
                    #calculate neuron activations
                    self.outputs[i] = self.pcnfwd(self.inputs[i],False)
                    #adjust weights
                    self.weights += self.eta*np.outer(np.transpose(self.inputs[i]),self.targets[i]-self.outputs[i])  
                    #print(n,'weights\n',self.weights)
   
        #Final Error
        self.outputs = self.pcnfwd(self.inputs,False)
        #print('Final Error',self.errfunc(self.outputs,self.targets))
     
        return 0

    def pcntrain_automatic(self,valid,validt,itSteps):
        """ train the perceptron until the error on the validation data increases """
        
        trainerror = np.array([])
        validerror = np.array([])

        n = -1    
        while n < 2 or (validerror[n] < validerror[n-1]):
            n+=1
            trainerror = np.append(trainerror,self.errfunc(self.pcnfwd(self.inputs,False),self.targets))
            validerror = np.append(validerror,self.errfunc(self.pcnfwd(valid,True),validt))
            self.pcntrain(itSteps)
        
        return trainerror,validerror

    def pcnfwd(self,inputs,addbias):
        """ run the network forward """
        
        #add biad node (optional)
        if addbias == True:
            inputs = np.concatenate((-np.ones((len(inputs),1)),inputs),axis=1) 
        
        if self.functype == 'linear':
            #linear function
            func = lambda x: x 
        elif self.functype == 'sigmoid':
            #sigmoid
            func = lambda x: 1/(1+np.exp(-x)) 
        elif self.functype == 'softmax':
            #softmax
            func = lambda x: np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)
            
        return func(np.dot(inputs,self.weights))    
    
    def confmat(self,inputs,targets):
        """ confusion matrix to evaluate the performance of the network """
        
        #output activations
        outputs = self.pcnfwd(inputs,True)	
        outputs = np.where(outputs>0.5,1,0)
        
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
        print('Confusion Matrix\n',cm)
        if np.sum(cm)!=0:
            print('Success Rate',100*np.trace(cm)/np.sum(cm),'%')
        return 0
            



